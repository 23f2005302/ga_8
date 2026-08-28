import json
import re
import unicodedata
import hashlib
import math
from datetime import datetime, timezone
import google_crc32c
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- GLOBAL STATE FOR /bqml ---
bqml_state = {}

@app.route('/', methods=['GET', 'POST'])
def health_check():
    return "Server is up and running!", 200

def parse_iso8601(dt_str):
    if type(dt_str) is not str: return False
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-](0[0-9]|1[0-3]):[0-5][0-9]|[+-]14:00)$'
    if not re.match(pattern, dt_str): return False
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception:
        return False

def norm_str(val):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val)).lower()).strip()

def extract_words(text):
    return set(re.findall(r'[^\W_]+', str(text), re.UNICODE))

# -----------------------------------------
# 1. NEW ENDPOINT: /bqml (Two-Phase Gate)
# -----------------------------------------
@app.route('/bqml', methods=['POST'], strict_slashes=False)
def bqml():
    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "INVALID_INPUT"}), 400

    if type(req) is not dict or req.get('phase') not in ['select', 'evaluate']:
        return jsonify({"error": "INVALID_INPUT"}), 400

    phase = req['phase']

    if phase == 'select':
        runId = req.get('runId')
        if type(runId) is not str or not runId or len(runId) > 128:
            return jsonify({
                "runId": runId if type(runId) is str else None,
                "selectedTrialId": None,
                "trainRowIds": [], "evalRowIds": [], "featureNames": [],
                "datasetDigest": None,
                "reasonCodes": ["INVALID_INPUT"]
            })

        if runId in bqml_state:
            if bqml_state[runId]['request'] != req:
                return jsonify({"error": "RUN_ID_CONFLICT"}), 409
            return jsonify(bqml_state[runId]['response'])

        reasonCodes = set()
        
        fb_features = req.get('forbiddenFeatures')
        limit = req.get('numTrialsLimit')
        rows = req.get('rows')
        trials = req.get('trials')

        if (type(fb_features) is not list or 
            type(limit) is not int or limit <= 0 or type(limit) is bool or
            type(rows) is not list or not rows or
            type(trials) is not list):
            reasonCodes.add("INVALID_INPUT")
        
        valid_rows = []
        if "INVALID_INPUT" not in reasonCodes:
            row_ids = set()
            for r in rows:
                if type(r) is not dict:
                    reasonCodes.add("INVALID_INPUT"); break
                rid = r.get('id'); ent = r.get('entity'); et = r.get('eventTime')
                pt = r.get('predictionTime'); ver = r.get('version'); sp = r.get('split')
                fts = r.get('features')
                
                if (type(rid) is not str or type(ent) is not str or type(et) is not str or 
                    type(pt) is not str or type(sp) is not str or type(fts) is not dict):
                    reasonCodes.add("INVALID_INPUT"); break
                    
                if type(ver) is not int or type(ver) is bool or ver < 0 or ver > 9007199254740991:
                    reasonCodes.add("INVALID_INPUT"); break
                
                if sp not in ["TRAIN", "EVAL"]:
                    reasonCodes.add("INVALID_INPUT"); break
                    
                dt_et = parse_iso8601(et)
                dt_pt = parse_iso8601(pt)
                if not dt_et or not dt_pt:
                    reasonCodes.add("INVALID_INPUT"); break
                
                if rid in row_ids:
                    reasonCodes.add("INVALID_INPUT"); break
                row_ids.add(rid)
                
                for fn, fv in fts.items():
                    if type(fv) is not dict or 'availableAt' not in fv:
                        reasonCodes.add("INVALID_INPUT"); break
                    if not parse_iso8601(fv['availableAt']):
                        reasonCodes.add("INVALID_INPUT"); break
                
                if "INVALID_INPUT" in reasonCodes: break
                
                r_copy = dict(r)
                r_copy['_dt_et'] = dt_et.astimezone(timezone.utc)
                r_copy['_dt_pt'] = dt_pt.astimezone(timezone.utc)
                valid_rows.append(r_copy)

        trial_ids = set()
        if "INVALID_INPUT" not in reasonCodes:
            for t in trials:
                if type(t) is not dict:
                    reasonCodes.add("INVALID_INPUT"); break
                tid = t.get('trialId'); st = t.get('status'); em = t.get('evalMetric')
                if type(tid) is not int or type(tid) is bool or tid < 0 or tid > 9007199254740991:
                    reasonCodes.add("INVALID_INPUT"); break
                if st not in ["SUCCEEDED", "FAILED"]:
                    reasonCodes.add("INVALID_INPUT"); break
                if type(em) not in (int, float) or type(em) is bool or not math.isfinite(em):
                    reasonCodes.add("INVALID_INPUT"); break
                
                if tid in trial_ids:
                    reasonCodes.add("INVALID_INPUT"); break
                trial_ids.add(tid)

            if len(trials) > limit and "INVALID_INPUT" not in reasonCodes:
                reasonCodes.add("TRIAL_LIMIT_EXCEEDED")

        train_ids = []
        eval_ids = []
        final_features = []
        best_trial = None

        if not reasonCodes:
            dedup = {}
            for r in valid_rows:
                k = (r['entity'], r['_dt_et'].strftime('%Y-%m-%dT%H:%M:%S.%f')[:23] + 'Z')
                if k not in dedup:
                    dedup[k] = r
                else:
                    exist = dedup[k]
                    if r['version'] > exist['version'] or (r['version'] == exist['version'] and r['id'].encode('utf-8') < exist['id'].encode('utf-8')):
                        dedup[k] = r
                        
            final_rows = list(dedup.values())
            all_fs = {}
            for r in final_rows:
                for fn, fv in r['features'].items():
                    if fn not in all_fs: all_fs[fn] = []
                    all_fs[fn].append({'pt': r['_dt_pt'], 'at': parse_iso8601(fv['availableAt']).astimezone(timezone.utc)})
            
            for fn, times in all_fs.items():
                if fn in fb_features: continue
                if len(times) != len(final_rows): continue
                if all(t['at'] <= t['pt'] for t in times):
                    final_features.append(fn)
                    
            train_r = [r['id'] for r in final_rows if r['split'] == "TRAIN"]
            eval_r = [r['id'] for r in final_rows if r['split'] == "EVAL"]
            
            train_r.sort(key=lambda x: x.encode('utf-8'))
            eval_r.sort(key=lambda x: x.encode('utf-8'))
            final_features.sort(key=lambda x: x.encode('utf-8'))
            
            train_ids = train_r
            eval_ids = eval_r
            
            succ = [t for t in trials if t['status'] == "SUCCEEDED"]
            if not succ:
                reasonCodes.add("NO_SUCCESSFUL_TRIAL")
            else:
                succ.sort(key=lambda x: (-x['evalMetric'], x['trialId']))
                best_trial = succ[0]['trialId']

        if reasonCodes:
            res = {
                "runId": runId, "selectedTrialId": None,
                "trainRowIds": [], "evalRowIds": [], "featureNames": [],
                "datasetDigest": None, "reasonCodes": sorted(list(reasonCodes), key=lambda x: x.encode('utf-8'))
            }
        else:
            digest_obj = {"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": final_features}
            digest_str = json.dumps(digest_obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            digest = hashlib.sha256(digest_str).hexdigest()
            res = {
                "runId": runId, "selectedTrialId": best_trial,
                "trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": final_features,
                "datasetDigest": digest, "reasonCodes": []
            }
            
        bqml_state[runId] = {'request': req, 'response': res}
        return jsonify(res)

    elif phase == 'evaluate':
        runId = req.get('runId')
        s_trial = req.get('selectedTrialId')
        d_digest = req.get('datasetDigest')
        m_floor = req.get('metricFloor')
        r_slices = req.get('requiredSlices')
        rows = req.get('rows')
        b_proc = req.get('bytesProcessed')
        m_bytes = req.get('maxBytes')
        
        reasonCodes = set()
        c_pass = True
        
        if (type(runId) is not str or type(s_trial) is not int or type(s_trial) is bool or 
            type(d_digest) is not str or type(m_floor) not in (int, float) or type(m_floor) is bool or 
            not (0 <= m_floor <= 1) or type(r_slices) is not dict or type(rows) is not list or 
            type(b_proc) is not int or type(b_proc) is bool or b_proc < 0 or
            type(m_bytes) is not int or type(m_bytes) is bool or m_bytes < 0):
            reasonCodes.add("INVALID_INPUT")
            c_pass = False
            
        if "INVALID_INPUT" not in reasonCodes:
            for k, v in r_slices.items():
                if type(k) is not str or type(v) not in (int, float) or type(v) is bool or not (0 <= v <= 1):
                    reasonCodes.add("INVALID_INPUT")
                    c_pass = False; break
                    
        lineage_ok = False
        if "INVALID_INPUT" not in reasonCodes:
            if runId in bqml_state:
                stored = bqml_state[runId]['response']
                if stored.get('selectedTrialId') == s_trial and stored.get('datasetDigest') == d_digest and s_trial is not None:
                    lineage_ok = True
            if not lineage_ok:
                reasonCodes.add("INVALID_LINEAGE")
                c_pass = False
                
        test_metric = None
        rows_ok = True
        
        if "INVALID_INPUT" not in reasonCodes:
            if not rows:
                rows_ok = False
            else:
                for r in rows:
                    if type(r) is not dict: rows_ok = False; break
                    lbl = r.get('label'); pr = r.get('prediction'); sl = r.get('slice')
                    if lbl not in [0, 1] or pr not in [0, 1] or type(sl) is not str or not sl:
                        rows_ok = False; break
                        
            if not rows_ok:
                reasonCodes.add("INVALID_TEST_ROW")
                c_pass = False
                
            if lineage_ok and rows_ok:
                agg_corr = sum(1 for r in rows if r['label'] == r['prediction'])
                test_metric = round(agg_corr / len(rows), 12) if len(rows) > 0 else 0.0
                if test_metric < m_floor:
                    reasonCodes.add("AGGREGATE_FLOOR")
                    
                slice_mets = {}
                for r in rows:
                    sl = r['slice']
                    if sl not in slice_mets: slice_mets[sl] = {'c': 0, 't': 0}
                    slice_mets[sl]['t'] += 1
                    if r['label'] == r['prediction']: slice_mets[sl]['c'] += 1
                    
                for req_s, floor in r_slices.items():
                    if req_s not in slice_mets:
                        reasonCodes.add(f"MISSING_SLICE:{req_s}")
                        c_pass = False
                    else:
                        sm = round(slice_mets[req_s]['c'] / slice_mets[req_s]['t'], 12)
                        if sm < floor:
                            reasonCodes.add(f"SLICE_FLOOR:{req_s}")
                            c_pass = False
                            
        if "INVALID_INPUT" not in reasonCodes:
            if b_proc > m_bytes:
                reasonCodes.add("BYTE_LIMIT")
                
        decision = "reject" if reasonCodes else "admit"
        
        return jsonify({
            "runId": runId if type(runId) is str else None,
            "selectedTrialId": s_trial if type(s_trial) is int and not isinstance(s_trial, bool) else None,
            "datasetDigest": d_digest if type(d_digest) is str else None,
            "testMetric": test_metric,
            "criticalSlicePass": c_pass,
            "decision": decision,
            "bytesProcessed": b_proc if type(b_proc) is int and not isinstance(b_proc, bool) else None,
            "reasonCodes": sorted(list(reasonCodes), key=lambda x: x.encode('utf-8'))
        })

# -----------------------------------------
# PREVIOUS ROUTES KEEP INTACT DOWN BELOW
# -----------------------------------------
@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict or type(req.get('policy')) is not dict or type(req.get('objects')) is not list: return jsonify({"error": "INVALID_INPUT"}), 400
    policy = req['policy']
    min_dt = parse_iso8601(policy.get('minTime'))
    max_dt = parse_iso8601(policy.get('maxTime'))
    thresh = policy.get('contaminationThreshold')
    policy_valid = False
    if min_dt and max_dt and type(thresh) in (int, float) and not isinstance(thresh, bool) and 0 <= thresh <= 1: policy_valid = True
    rej_objs, rej_rows, lineage, valid_rows = [], [], [], []
    for item in req['objects']:
        obj = item if type(item) is dict else {}
        obj_codes = set()
        uri = obj.get('uri')
        if type(uri) is not str or uri != "gs://bucket/object": obj_codes.add("URI_INVALID")
        gen, f_gen = obj.get('generation'), obj.get('fetchedGeneration')
        gen_valid = type(gen) is str and gen.isdecimal()
        f_gen_valid = type(f_gen) is str and f_gen.isdecimal()
        if not gen_valid or not f_gen_valid: obj_codes.add("GENERATION_INVALID")
        if gen != f_gen: obj_codes.add("GENERATION_MISMATCH")
        crc_in = obj.get('crc32c')
        crc_syntax_valid = type(crc_in) is str and bool(re.match(r'^[a-f0-9]{8}$', crc_in))
        if not crc_syntax_valid: obj_codes.add("CRC32C_INVALID")
        content = obj.get('content')
        if type(content) is not str: obj_codes.add("SCHEMA_INVALID")
        else:
            if crc_syntax_valid:
                calc_crc = google_crc32c.Checksum(content.encode('utf-8')).hexdigest().decode('utf-8').lower().zfill(8)
                if calc_crc != crc_in: obj_codes.add("CRC32C_MISMATCH")
        schema_id = obj.get('schemaId')
        if schema_id != 'training-v1': obj_codes.add("SCHEMA_INVALID")
        obj_rows = []
        if type(content) is str:
            lines = content.split('\n')
            has_non_blank = False
            for line in lines:
                if not line.strip(): continue
                has_non_blank = True
                try: row = json.loads(line)
                except ValueError: obj_codes.add("JSONL_INVALID"); continue
                if type(row) is not dict or set(row.keys()) != {'id', 'entity', 'eventTime', 'revision', 'text'}: obj_codes.add("SCHEMA_INVALID"); continue
                if type(row['id']) is not str or type(row['entity']) is not str or type(row['eventTime']) is not str or type(row['text']) is not str: obj_codes.add("SCHEMA_INVALID"); continue
                rev = row['revision']
                if type(rev) is not int or isinstance(rev, bool) or rev < 0 or rev > 9007199254740991: obj_codes.add("SCHEMA_INVALID"); continue
                dt = parse_iso8601(row['eventTime'])
                if not dt: obj_codes.add("SCHEMA_INVALID"); continue
                r_c = {"id": row['id'], "entity": norm_str(row['entity']), "eventTime": dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:23] + 'Z', "revision": rev, "text": norm_str(row['text'])}
                r_c['words'] = extract_words(r_c['text'])
                obj_rows.append(r_c)
            if not has_non_blank: obj_codes.add("SCHEMA_INVALID")
        if obj_codes: rej_objs.append({"uri": uri if type(uri) is str else None, "reasonCodes": sorted(list(obj_codes), key=lambda x: x.encode('utf-8'))})
        else:
            lineage.append({"uri": uri, "generation": gen, "crc32c": crc_in, "schemaId": schema_id})
            valid_rows.extend(obj_rows)
    dedup = {}
    for r in valid_rows:
        key = (r['entity'], r['eventTime'], r['text'])
        if key not in dedup: dedup[key] = r
        else:
            exist = dedup[key]
            r_id_bytes, exist_id_bytes = r['id'].encode('utf-8'), exist['id'].encode('utf-8')
            if r['revision'] > exist['revision'] or (r['revision'] == exist['revision'] and r_id_bytes < exist_id_bytes):
                rej_rows.append({"id": exist['id'], "reasonCodes": ["DUPLICATE"]})
                dedup[key] = r
            else: rej_rows.append({"id": r['id'], "reasonCodes": ["DUPLICATE"]})
    splits = {"train": [], "validation": [], "test": []}
    train_words_list = []
    for r in dedup.values():
        if not policy_valid: rej_rows.append({"id": r['id'], "reasonCodes": ["POLICY_INVALID"]}); continue
        r_dt = datetime.fromisoformat(r['eventTime'].replace('Z', '+00:00'))
        if not (min_dt <= r_dt <= max_dt): rej_rows.append({"id": r['id'], "reasonCodes": ["OUT_OF_WINDOW"]}); continue
        bucket = hashlib.sha256(r['entity'].encode('utf-8')).digest()[0] % 10
        if bucket <= 5: splits['train'].append(r); train_words_list.append(r['words'])
        elif bucket <= 7: splits['validation'].append(r)
        else: splits['test'].append(r)
    for s_name in ['validation', 'test']:
        clean = []
        for r in splits[s_name]:
            contaminated = False
            if train_words_list:
                s1 = r['words']
                for tw in train_words_list:
                    sim = 1.0 if not s1 and not tw else (0.0 if not s1 or not tw else len(s1 & tw) / float(len(s1 | tw)))
                    if sim >= thresh: contaminated = True; break
            if contaminated: rej_rows.append({"id": r['id'], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else: clean.append(r)
        splits[s_name] = clean
    digests, final_splits = {}, {}
    def dict_sort_key(x):
        u = x.get('uri') if 'uri' in x else x.get('id')
        k1 = u.encode('utf-8') if type(u) is str else b''
        k2 = json.dumps(x, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return (k1, k2)
    for s_name in ['train', 'validation', 'test']:
        splits[s_name].sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps({"id": x['id'], "entity": x['entity'], "eventTime": x['eventTime'], "revision": x['revision'], "text": x['text']}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')))
        ordered_lines, final_list = [], []
        for r in splits[s_name]:
            ordered = {"id": r['id'], "entity": r['entity'], "eventTime": r['eventTime'], "revision": r['revision'], "text": r['text']}
            final_list.append(ordered)
            ordered_lines.append(json.dumps(ordered, separators=(',', ':'), ensure_ascii=False))
        out_bytes = "".join(l + "\n" for l in ordered_lines).encode('utf-8') if ordered_lines else b""
        digests[s_name] = hashlib.sha256(out_bytes).hexdigest()
        final_splits[s_name] = final_list
    for row in rej_rows:
        if 'reasonCodes' in row: row['reasonCodes'] = sorted(list(set(row['reasonCodes'])), key=lambda x: x.encode('utf-8'))
    rej_objs.sort(key=dict_sort_key); rej_rows.sort(key=dict_sort_key); lineage.sort(key=dict_sort_key)
    return jsonify({"splits": final_splits, "rejectedObjects": rej_objs, "rejectedRows": rej_rows, "digests": digests, "lineage": lineage})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
