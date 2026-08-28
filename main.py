import json, re, unicodedata, hashlib
from datetime import datetime, timezone
import google_crc32c
from flask import request, jsonify

def is_valid_iso8601(dt_str):
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-](0[0-9]|1[0-3]):[0-5][0-9]|14:00)$'
    if not re.match(pattern, str(dt_str)): return False
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt
    except ValueError:
        return False

def norm_str(val):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val)).lower()).strip()

def extract_words(text):
    return set(re.findall(r'[^\W_]+', str(text), re.UNICODE))

def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)

@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    req = request.get_json(force=True, silent=True)
    if not isinstance(req, dict) or 'policy' not in req or not isinstance(req.get('objects'), list):
        return jsonify({"error": "INVALID_INPUT"}), 400

    policy = req['policy']
    min_dt = is_valid_iso8601(policy.get('minTime'))
    max_dt = is_valid_iso8601(policy.get('maxTime'))
    thresh = policy.get('contaminationThreshold')
    policy_valid = min_dt and max_dt and isinstance(thresh, (int, float)) and 0 <= thresh <= 1

    rej_objs, rej_rows, lineage = [], [], []
    valid_rows = []

    for obj in req['objects']:
        if not isinstance(obj, dict): continue
        uri = obj.get('uri')
        obj_codes = []
        
        if not isinstance(uri, str): obj_codes.append("URI_INVALID")
        elif uri != "gs://bucket/object": obj_codes.append("URI_INVALID")
            
        gen, f_gen = obj.get('generation'), obj.get('fetchedGeneration')
        if not isinstance(gen, str) or not re.match(r'^\d+$', gen): obj_codes.append("GENERATION_INVALID")
        elif gen != f_gen: obj_codes.append("GENERATION_MISMATCH")

        crc_in = obj.get('crc32c')
        if not isinstance(crc_in, str) or not re.match(r'^[a-f0-9]{8}$', crc_in):
            obj_codes.append("CRC32C_INVALID")
        
        content = obj.get('content')
        if not isinstance(content, str):
            obj_codes.append("SCHEMA_INVALID")
        else:
            if "CRC32C_INVALID" not in obj_codes:
                calc_crc = f"{google_crc32c.Checksum(content.encode('utf-8')).hexdigest().decode('utf-8')}"
                if calc_crc != crc_in: obj_codes.append("CRC32C_MISMATCH")

        if obj.get('schemaId') != 'training-v1':
            obj_codes.append("SCHEMA_INVALID")

        lines = [l for l in str(content).split('\n') if l.strip()]
        if not lines and "SCHEMA_INVALID" not in obj_codes:
            obj_codes.append("SCHEMA_INVALID")

        obj_rows = []
        for line in lines:
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row.keys()) != {'id', 'entity', 'eventTime', 'revision', 'text'}:
                    obj_codes.append("SCHEMA_INVALID")
                    continue
                if not all(isinstance(row[k], str) for k in ['id', 'entity', 'text']) or not isinstance(row['revision'], int) or row['revision'] < 0:
                    obj_codes.append("SCHEMA_INVALID")
                    continue
                dt = is_valid_iso8601(row['eventTime'])
                if not dt:
                    obj_codes.append("SCHEMA_INVALID")
                    continue
                
                # Normalization
                row['entity_norm'] = norm_str(row['entity'])
                row['text_norm'] = norm_str(row['text'])
                row['eventTime_norm'] = dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                row['words'] = extract_words(row['text_norm'])
                obj_rows.append(row)
            except json.JSONDecodeError:
                obj_codes.append("JSONL_INVALID")

        if obj_codes:
            rej_objs.append({"uri": uri if isinstance(uri, str) else None, "reasonCodes": sorted(list(set(obj_codes)))})
        else:
            lineage.append({"uri": uri, "generation": gen, "crc32c": crc_in, "schemaId": obj.get('schemaId')})
            valid_rows.extend(obj_rows)

    # Deduplication
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

    # Contamination Check
    for s_name in ['validation', 'test']:
        clean = []
        for r in splits[s_name]:
            contaminated = any(jaccard(r['words'], tw) >= thresh for tw in train_words)
            if contaminated:
                rej_rows.append({"id": r['id'], "reasonCodes": ["TRAIN_CONTAMINATION"]})
            else:
                clean.append(r)
        splits[s_name] = clean

    # Formatting and Hashing
    digests = {}
    final_splits = {}
    for s_name in ['train', 'validation', 'test']:
        splits[s_name].sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))
        lines = []
        for r in splits[s_name]:
            lines.append(json.dumps({"id": r['id'], "entity": r['entity'], "eventTime": r['eventTime_norm'], "revision": r['revision'], "text": r['text']}, separators=(',', ':'), ensure_ascii=False))
        out_str = "".join(l + "\n" for l in lines).encode('utf-8')
        digests[s_name] = hashlib.sha256(out_str).hexdigest()
        final_splits[s_name] = [json.loads(l) for l in lines] # Output requires array of objects

    rej_objs.sort(key=lambda x: (str(x['uri']).encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))
    rej_rows.sort(key=lambda x: (x['id'].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))
    lineage.sort(key=lambda x: (x['uri'].encode('utf-8'), json.dumps(x, separators=(',', ':')).encode('utf-8')))

    return jsonify({
        "splits": final_splits,
        "rejectedObjects": rej_objs,
        "rejectedRows": rej_rows,
        "digests": digests,
        "lineage": lineage
    })
