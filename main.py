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

def is_valid_iso8601(dt_str):
    if not isinstance(dt_str, str): return False
    # Strict YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm) with max offset ±14:00
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-](0[0-9]|1[0-3]):[0-5][0-9]|[+-]14:00)$'
    if not re.match(pattern, dt_str): return False
    try:
        if dt_str.endswith('Z'):
            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(dt_str)
        return dt
    except ValueError:
        return False

def norm_str(val):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val)).lower()).strip()

def extract_words(text):
    # Extracts words (letters and numbers only)
    return set(re.findall(r'[^\W_]+', str(text), re.UNICODE))

def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / float(len(s1 | s2))

@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    req = request.get_json(force=True, silent=True)
    if not isinstance(req, dict) or 'policy' not in req or not isinstance(req.get('objects'), list):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = req.get('policy', {})
    min_dt = is_valid_iso8601(policy.get('minTime'))
    max_dt = is_valid_iso8601(policy.get('maxTime'))
    thresh = policy.get('contaminationThreshold')
    policy_valid = bool(min_dt and max_dt and isinstance(thresh, (int, float)) and not isinstance(thresh, bool) and 0 <= thresh <= 1)

    rej_objs, rej_rows, lineage = [], [], []
    valid_rows = []

    for obj in req.get('objects', []):
        if not isinstance(obj, dict): continue
        uri = obj.get('uri')
        obj_codes = set()
        
        # 1. URI check
        if not isinstance(uri, str) or uri != "gs://bucket/object":
            obj_codes.add("URI_INVALID")
            
        # 2. Generation checks
        gen = obj.get('generation')
        f_gen = obj.get('fetchedGeneration')
        if not isinstance(gen, str) or not re.match(r'^\d+$', gen):
            obj_codes.add("GENERATION_INVALID")
        if gen != f_gen:
            obj_codes.add("GENERATION_MISMATCH")

        # 3. CRC32C checks
        crc_in = obj.get('crc32c')
        crc_invalid = False
        if not isinstance(crc_in, str) or not re.match(r'^[a-f0-9]{8}$', crc_in):
            obj_codes.add("CRC32C_INVALID")
            crc_invalid = True
        
        # 4. Content schema checks
        content = obj.get('content')
        if not isinstance(content, str):
            obj_codes.add("SCHEMA_INVALID")
        else:
            if not crc_invalid:
                calc_crc = google_crc32c.Checksum(content.encode('utf-8')).hexdigest().decode('utf-8')
                if calc_crc != crc_in:
                    obj_codes.add("CRC32C_MISMATCH")

        schema_id = obj.get('schemaId')
        if schema_id != 'training-v1':
            obj_codes.add("SCHEMA_INVALID")

        # 5. Row-level JSONL parsing and schema
        obj_rows = []
        if isinstance(content, str):
            lines = content.split('\n')
            valid_line_count = 0
            for line in lines:
                if not line.strip(): continue
                valid_line_count += 1
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
                # Strict check for non-negative safe integer (bools inherit int in Python, so exclude them)
                if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0 or rev > 9007199254740991:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                    
                dt = is_valid_iso8601(row['eventTime'])
                if not dt:
                    obj_codes.add("SCHEMA_INVALID")
                    continue
                
                # Copy row for canonicalized storage
                r_c = dict(row)
                r_c['entity_norm'] = norm_str(row['entity'])
                r_c['text_norm'] = norm_str(row['text'])
                r_c['eventTime_norm'] = dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                r_c['words'] = extract_words(r_c['text_norm'])
                obj_rows.append(r_c)

            if valid_line_count == 0:
                obj_codes.add("SCHEMA_INVALID")

        # Append to rejectedObjects or Lineage
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

    # 6. Tie-breaker Deduplication
    dedup = {}
    for r in valid_rows:
        key = (r['entity_norm'], r['eventTime_norm'], r['text_norm'])
        if key not in dedup:
            dedup[key] = r
        else:
            exist = dedup[key]
            if r['revision'] > exist['revision'] or (r['revision'] == exist['revision'] and r['id'].encode('utf-8') < exist['id'].encode('utf-8')):
                rej_rows.append({"id": exist['id'], "reasonCodes": ["DUPLICATE"]})
                dedup[key] = r
            else:
                rej_rows.append({"id": r['id'], "reasonCodes": ["DUPLICATE"]})

    splits = {"train": [], "validation": [], "test": []}
    train_words = []

    for key, r in dedup.items():
        if not policy_valid:
            rej_rows.append({"id": r['id'], "reasonCodes": ["POLICY_INVALID"]})
            continue
        
        r_dt = datetime.fromisoformat(r['eventTime_norm'].replace('Z', '+00:00'))
        if not (min_dt <= r_dt <= max_dt):
            rej_rows.append({"id": r['id'], "reasonCodes": ["OUT_OF_WINDOW"]})
            continue

        bucket = hashlib.sha256(r['entity_norm'].encode('utf-8')).digest()[0] % 10
        if bucket <= 5:
            splits['train'].append(r)
            train_words.append(r['words'])
        elif bucket <= 7:
            splits['validation'].append(r)
        else:
            splits['test'].append(r)

    # 7. Contamination Check
    for s_name in ['validation', 'test']:
        clean = []
        for r in splits[s_name]:
            contaminated = any(jaccard(r['words'], tw) >= thresh for tw in train_words)
            if contaminated:
                rej_rows.append({"id": r['id'], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                clean.append(r)
        splits[s_name] = clean

    # 8. Tie-breaker Sort & Format for response
    digests = {}
    final_splits = {}
    
    def obj_sort_key(x):
        u = x.get('uri')
        return (u.encode('utf-8') if isinstance(u, str) else b'', json.dumps(x, separators=(',', ':')).encode('utf-8'))
        
    def row_sort_key(x):
        i = x.get('id')
        return (i.encode('utf-8') if isinstance(i, str) else b'', json.dumps(x, separators=(',', ':')).encode('utf-8'))

    for s_name in ['train', 'validation', 'test']:
        splits[s_name].sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))
        ordered_lines = []
        final_list = []
        for r in splits[s_name]:
            # Serialize split rows in exact key order: id, entity, eventTime, revision, text
            ordered = {"id": r['id'], "entity": r['entity'], "eventTime": r['eventTime_norm'], "revision": r['revision'], "text": r['text']}
            final_list.append(ordered)
            ordered_lines.append(json.dumps(ordered, separators=(',', ':'), ensure_ascii=False))
            
        out_bytes = "".join(l + "\n" for l in ordered_lines).encode('utf-8')
        digests[s_name] = hashlib.sha256(out_bytes).hexdigest()
        final_splits[s_name] = final_list

    rej_objs.sort(key=obj_sort_key)
    rej_rows.sort(key=row_sort_key)
    lineage.sort(key=obj_sort_key)

    return jsonify({
        "splits": final_splits,
        "rejectedObjects": rej_objs,
        "rejectedRows": rej_rows,
        "digests": digests,
        "lineage": lineage
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
