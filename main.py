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
    if not isinstance(dt_str, str): return False
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-](0[0-9]|1[0-3]):[0-5][0-9]|[+-]14:00)$'
    if not re.match(pattern, dt_str): return False
    try:
        if dt_str.endswith('Z'):
            return datetime.fromisoformat(dt_str[:-1] + '+00:00')
        return datetime.fromisoformat(dt_str)
    except Exception:
        return False

def norm_str(val):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val)).lower()).strip()

def extract_words(text):
    return set(re.findall(r'[^\W_]+', str(text), re.UNICODE))

@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    req = request.get_json(force=True, silent=True)
    if req is None or 'policy' not in req or not isinstance(req.get('objects'), list):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = req['policy']
    min_dt = None
    max_dt = None
    thresh = None
    policy_valid = False

    if isinstance(policy, dict):
        min_dt = parse_iso8601(policy.get('minTime'))
        max_dt = parse_iso8601(policy.get('maxTime'))
        thresh = policy.get('contaminationThreshold')
        if min_dt and max_dt and isinstance(thresh, (int, float)) and not isinstance(thresh, bool) and 0 <= thresh <= 1:
            policy_valid = True

    rej_objs = []
    rej_rows = []
    lineage = []
    valid_rows = []

    for obj in req['objects']:
        if not isinstance(obj, dict): continue
        uri = obj.get('uri')
        obj_codes = set()
        
        # 1. URI Check
        if not isinstance(uri, str) or uri != "gs://bucket/object":
            obj_codes.add("URI_INVALID")
            
        # 2. Generation Checks (Both must be valid decimal strings)
        gen = obj.get('generation')
        f_gen = obj.get('fetchedGeneration')
        gen_is_valid = isinstance(gen, str) and bool(re.match(r'^\d+$', gen))
        f_gen_is_valid = isinstance(f_gen, str) and bool(re.match(r'^\d+$', f_gen))
        
        if not gen_is_valid or not f_gen_is_valid:
            obj_codes.add("GENERATION_INVALID")
        if gen != f_gen:
            obj_codes.add("GENERATION_MISMATCH")

        # 3. CRC32C Checks (Strict 8 lowercase hex digits)
        crc_in = obj.get('crc32c')
        crc_syntax_valid = isinstance(crc_in, str) and bool(re.match(r'^[a-f0-9]{8}$', crc_in))
        if not crc_syntax_valid:
            obj_codes.add("CRC32C_INVALID")
        
        # 4. Schema Checks
        content = obj.get('content')
        if not isinstance(content, str):
            obj_codes.add("SCHEMA_INVALID")
        else:
            if crc_syntax_valid:
                c = google_crc32c.Checksum(content.encode('utf-8'))
                calc_crc = c.hexdigest().decode('utf-8').lower().zfill(8)
                if calc_crc != crc_in:
                    obj_codes.add("CRC32C_MISMATCH")

        schema_id = obj.get('schemaId')
        if schema_id != 'training-v1':
            obj_codes.add("SCHEMA_INVALID")

        # 5. Row Validations
        obj_rows = []
        if isinstance(content, str):
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
                    
                if not isinstance(row, dict) or set(row.keys()) != {'id', 'entity', 'eventTime', 'revision', 'text'}:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                if not all(isinstance(row[k], str) for k in ['id', 'entity', 'eventTime', 'text']):
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                rev = row.get('revision')
                if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0 or rev > 9007199254740991:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                dt = parse_iso8601(row['eventTime'])
                if not dt:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                
                r_c = dict(row)
                r_c['entity_norm'] = norm_str(row['entity'])
                r_c['text_norm'] = norm_str(row['text'])
                r_c['eventTime_norm'] = dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                r_c['words'] = extract_words(r_c['text_norm'])
                obj_rows.append(r_c)

            # Check for totally empty file
            if not has_non_blank:
                obj_codes.add("SCHEMA_INVALID")

        if obj_codes:
            rej_objs.append({
                "uri": uri if isinstance(uri, str) else None,
                "reasonCodes": sorted(list(obj_codes))
            })
        else:
            lineage.append({
                "uri": uri, "generation": gen, "crc32c": crc_in, "schemaId": schema_id
            })
            valid_rows.extend(obj_rows)

    # 6. Deduplication
    dedup = {}
    for r in valid_rows:
        key = (r['entity_norm'], r['eventTime_norm'], r['text_norm'])
        if key not in dedup:
            dedup[key] = r
        else:
            exist = dedup[key]
            r_id_bytes = r['id'].encode('utf-8')
            exist_id_bytes = exist['id'].encode('utf-8')
            if r['revision'] > exist['revision'] or (r['revision'] == exist['revision'] and r_id_bytes < exist_id_bytes):
                rej_rows.append({"id": exist['id'], "reasonCodes": ["DUPLICATE"]})
                dedup[key] = r
            else:
                rej_rows.append({"id": r['id'], "reasonCodes": ["DUPLICATE"]})

    # 7. Splits & Contamination
    splits = {"train": [], "validation": [], "test": []}
    train_words_list = []

    for r in dedup.values():
        if not policy_valid:
            rej_rows.append({"id": r['id'], "reasonCodes": ["POLICY_INVALID"]})
            continue
        
        r_dt = parse_iso8601(r['eventTime_norm'])
        if not (min_dt <= r_dt <= max_dt):
            rej_rows.append({"id": r['id'], "reasonCodes": ["OUT_OF_WINDOW"]})
            continue

        bucket = hashlib.sha256(r['entity_norm'].encode('utf-8')).digest()[0] % 10
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
            for tw in train_words_list:
                s1, s2 = r['words'], tw
                sim = 1.0 if not s1 and not s2 else (0.0 if not s1 or not s2 else len(s1 & s2) / float(len(s1 | s2)))
                if sim >= thresh:
                    contaminated = True
                    break
                    
            if contaminated:
                rej_rows.append({"id": r['id'], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                clean.append(r)
        splits[s_name] = clean

    # 8. Exact Deterministic Formatting
    digests = {}
    final_splits = {}
    
    def dict_sort_key(x):
        u = x.get('uri') if 'uri' in x else x.get('id')
        k1 = u.encode('utf-8') if isinstance(u, str) else b''
        k2 = json.dumps(x, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        return (k1, k2)

    for s_name in ['train', 'validation', 'test']:
        splits[s_name].sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps({"id": x['id'], "entity": x['entity'], "eventTime": x['eventTime_norm'], "revision": x['revision'], "text": x['text']}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')))
        ordered_lines = []
        final_list = []
        for r in splits[s_name]:
            ordered = {"id": r['id'], "entity": r['entity'], "eventTime": r['eventTime_norm'], "revision": r['revision'], "text": r['text']}
            final_list.append(ordered)
            ordered_lines.append(json.dumps(ordered, separators=(',', ':'), ensure_ascii=False))
            
        out_bytes = "".join(l + "\n" for l in ordered_lines).encode('utf-8')
        digests[s_name] = hashlib.sha256(out_bytes).hexdigest()
        final_splits[s_name] = final_list

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
