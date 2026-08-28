import json
import re
import unicodedata
import hashlib
from datetime import datetime, timezone
import google_crc32c
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "INVALID_INPUT"}), 400

    if type(req) is not dict or type(req.get('policy')) is not dict or type(req.get('objects')) is not list:
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = req['policy']
    min_dt = parse_iso8601(policy.get('minTime'))
    max_dt = parse_iso8601(policy.get('maxTime'))
    thresh = policy.get('contaminationThreshold')
    
    policy_valid = False
    if min_dt and max_dt and type(thresh) in (int, float) and not isinstance(thresh, bool) and 0 <= thresh <= 1:
        policy_valid = True

    rej_objs, rej_rows, lineage, valid_rows = [], [], [], []

    for item in req['objects']:
        obj = item if type(item) is dict else {}
        obj_codes = set()
        
        uri = obj.get('uri')
        if type(uri) is not str or uri != "gs://bucket/object":
            obj_codes.add("URI_INVALID")
            
        gen, f_gen = obj.get('generation'), obj.get('fetchedGeneration')
        gen_valid = type(gen) is str and gen.isdecimal()
        f_gen_valid = type(f_gen) is str and f_gen.isdecimal()
        
        if not gen_valid or not f_gen_valid: obj_codes.add("GENERATION_INVALID")
        if gen != f_gen: obj_codes.add("GENERATION_MISMATCH")

        crc_in = obj.get('crc32c')
        crc_syntax_valid = type(crc_in) is str and bool(re.match(r'^[a-f0-9]{8}$', crc_in))
        if not crc_syntax_valid:
            obj_codes.add("CRC32C_INVALID")
        
        content = obj.get('content')
        if type(content) is not str:
            obj_codes.add("SCHEMA_INVALID")
        else:
            if crc_syntax_valid:
                calc_crc = google_crc32c.Checksum(content.encode('utf-8')).hexdigest().decode('utf-8').lower().zfill(8)
                if calc_crc != crc_in:
                    obj_codes.add("CRC32C_MISMATCH")

        schema_id = obj.get('schemaId')
        if schema_id != 'training-v1':
            obj_codes.add("SCHEMA_INVALID")

        obj_rows = []
        if type(content) is str:
            lines = content.split('\n')
            has_non_blank = False
            for line in lines:
                if not line.strip(): continue
                has_non_blank = True
                try:
                    row = json.loads(line)
                except ValueError:
                    obj_codes.add("JSONL_INVALID")
                    continue
                    
                if type(row) is not dict or set(row.keys()) != {'id', 'entity', 'eventTime', 'revision', 'text'}:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                if type(row['id']) is not str or type(row['entity']) is not str or type(row['eventTime']) is not str or type(row['text']) is not str:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                rev = row['revision']
                if type(rev) is not int or isinstance(rev, bool) or rev < 0 or rev > 9007199254740991:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                dt = parse_iso8601(row['eventTime'])
                if not dt:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                
                # We canonicalize values and store them immediately so they are used in the final response
                r_c = {
                    "id": row['id'],
                    "entity": norm_str(row['entity']),
                    "eventTime": dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:23] + 'Z',
                    "revision": rev,
                    "text": norm_str(row['text'])
                }
                r_c['words'] = extract_words(r_c['text'])
                obj_rows.append(r_c)

            if not has_non_blank:
                obj_codes.add("SCHEMA_INVALID")

        if obj_codes:
            rej_objs.append({
                "uri": uri if type(uri) is str else None,
                "reasonCodes": sorted(list(obj_codes), key=lambda x: x.encode('utf-8'))
            })
        else:
            lineage.append({"uri": uri, "generation": gen, "crc32c": crc_in, "schemaId": schema_id})
            valid_rows.extend(obj_rows)

    dedup = {}
    for r in valid_rows:
        key = (r['entity'], r['eventTime'], r['text'])
        if key not in dedup:
            dedup[key] = r
        else:
            exist = dedup[key]
            r_id_bytes, exist_id_bytes = r['id'].encode('utf-8'), exist['id'].encode('utf-8')
            if r['revision'] > exist['revision'] or (r['revision'] == exist['revision'] and r_id_bytes < exist_id_bytes):
                rej_rows.append({"id": exist['id'], "reasonCodes": ["DUPLICATE"]})
                dedup[key] = r
            else:
                rej_rows.append({"id": r['id'], "reasonCodes": ["DUPLICATE"]})

    splits = {"train": [], "validation": [], "test": []}
    train_words_list = []

    for r in dedup.values():
        if not policy_valid:
            rej_rows.append({"id": r['id'], "reasonCodes": ["POLICY_INVALID"]})
            continue
        
        r_dt = datetime.fromisoformat(r['eventTime'].replace('Z', '+00:00'))
        if not (min_dt <= r_dt <= max_dt):
            rej_rows.append({"id": r['id'], "reasonCodes": ["OUT_OF_WINDOW"]})
            continue

        bucket = hashlib.sha256(r['entity'].encode('utf-8')).digest()[0] % 10
        if bucket <= 5:
            splits['train'].append(r)
            train_words_list.append(r['words'])
        elif bucket <= 7:
            splits['validation'].append(r)
        else:
            splits['test'].append(r)

    for s_name in ['validation', 'test']:
        clean = []
        for r in splits[s_name]:
            contaminated = False
            if train_words_list: 
                s1 = r['words']
                for tw in train_words_list:
                    sim = 1.0 if not s1 and not tw else (0.0 if not s1 or not tw else len(s1 & tw) / float(len(s1 | tw)))
                    if sim >= thresh:
                        contaminated = True
                        break
            if contaminated:
                rej_rows.append({"id": r['id'], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                clean.append(r)
        splits[s_name] = clean

    digests = {}
    final_splits = {}
    
    def dict_sort_key(x):
        u = x.get('uri') if 'uri' in x else x.get('id')
        k1 = u.encode('utf-8') if type(u) is str else b''
        k2 = json.dumps(x, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return (k1, k2)

    for s_name in ['train', 'validation', 'test']:
        splits[s_name].sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps({"id": x['id'], "entity": x['entity'], "eventTime": x['eventTime'], "revision": x['revision'], "text": x['text']}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')))
        ordered_lines = []
        final_list = []
        for r in splits[s_name]:
            ordered = {"id": r['id'], "entity": r['entity'], "eventTime": r['eventTime'], "revision": r['revision'], "text": r['text']}
            final_list.append(ordered)
            ordered_lines.append(json.dumps(ordered, separators=(',', ':'), ensure_ascii=False))
            
        out_bytes = "".join(l + "\n" for l in ordered_lines).encode('utf-8') if ordered_lines else b""
        digests[s_name] = hashlib.sha256(out_bytes).hexdigest()
        final_splits[s_name] = final_list

    for row in rej_rows:
        if 'reasonCodes' in row:
            row['reasonCodes'] = sorted(list(set(row['reasonCodes'])), key=lambda x: x.encode('utf-8'))

    rej_objs.sort(key=dict_sort_key)
    rej_rows.sort(key=dict_sort_key)
    lineage.sort(key=dict_sort_key)

    return jsonify({
        "splits": final_splits,
        "rejectedObjects": rej_objs,
        "rejectedRows": rej_rows,
        "digests": digests,
        "lineage": lineage
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
