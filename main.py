import json
import re
import unicodedata
import hashlib
import math
import urllib.parse
import html
from datetime import datetime, timezone
import google_crc32c
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- GLOBAL STATES ---
bqml_state = {}
quantize_state = {}
pipeline_state = {}
DAG_NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]

# --- HELPER FUNCTIONS ---
def parse_iso8601(dt_str):
    if type(dt_str) is not str: return False
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-](0[0-9]|1[0-3]):[0-5][0-9]|[+-]14:00)$'
    if not re.match(pattern, dt_str): return False
    try: return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except Exception: return False

def norm_str(val):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(val)).lower()).strip()

def extract_words(text):
    return set(re.findall(r'[^\W_]+', str(text), re.UNICODE))

def is_safe_int(x):
    return type(x) is int and not isinstance(x, bool) and 0 <= x <= 9007199254740991

def is_safe_pos_int(x):
    return type(x) is int and not isinstance(x, bool) and 1 <= x <= 9007199254740991

def is_finite_num(x):
    return type(x) in (int, float) and not isinstance(x, bool) and math.isfinite(x)

def check_channel_rules(channel, output):
    if channel == 'html':
        if re.search(r'<\s*(script|iframe|object|embed)\b', output, re.IGNORECASE): return "SCRIPT_TAG"
        if re.search(r'\bon[a-zA-Z]+\s*=', output, re.IGNORECASE): return "EVENT_HANDLER"
    if channel == 'sql':
        if re.search(r'[\'";]|--|/\*|\bunion\b|\bor\s+1\s*=\s*1\b', output, re.IGNORECASE): return "SQL_METACHAR"
    if channel == 'shell':
        if re.search(r'[;&|`<>]|\$\(|\$\{', output): return "SHELL_METACHAR"
    if channel in ['html', 'markdown', 'url']:
        if re.search(r'(javascript|data|vbscript)\s*:', output, re.IGNORECASE): return "DANGEROUS_SCHEME"
        urls = []
        if channel == 'html': urls = re.findall(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', output, re.IGNORECASE)
        elif channel == 'markdown': urls = re.findall(r'\]\(([^)]+)\)', output)
        elif channel == 'url': urls = [output.strip()]
        allowed_hosts = {'cdn-e93c2c6.example', 'app-zttqqmn.example'}
        for u in urls:
            if u.startswith('//'): u = 'https:' + u
            parsed = urllib.parse.urlparse(u)
            if parsed.scheme and parsed.scheme.lower() not in ['http', 'https']: return "DANGEROUS_SCHEME"
            if parsed.hostname and parsed.hostname.lower() not in allowed_hosts: return "EXTERNAL_EXFIL"
    return "SAFE"

def canon_json(data):
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)

def sha256_canon(data):
    return hashlib.sha256(canon_json(data).encode('utf-8')).hexdigest()

def compute_keys(inputs, cache):
    keys = {}
    k_vd = sha256_canon([inputs.get("generation"), inputs.get("checksum")])
    keys["verify_data"] = k_vd
    k_prep = sha256_canon([inputs.get("canonicalData"), inputs.get("prepareCode"), inputs.get("prepareConfig")]) if k_vd and k_vd in cache.get("verify_data", {}) else None
    keys["prepare"] = k_prep
    k_train = sha256_canon([cache["prepare"][k_prep]["artifactDigest"], inputs.get("trainCode"), inputs.get("trainConfig"), inputs.get("runtime")]) if k_prep and k_prep in cache.get("prepare", {}) else None
    keys["train"] = k_train
    k_eval = sha256_canon([cache["train"][k_train]["artifactDigest"], inputs.get("canonicalData"), inputs.get("evaluateCode"), inputs.get("evaluateConfig")]) if k_train and k_train in cache.get("train", {}) else None
    keys["evaluate"] = k_eval
    k_reg = sha256_canon([cache["evaluate"][k_eval]["artifactDigest"], inputs.get("schemaDigest")]) if k_eval and k_eval in cache.get("evaluate", {}) else None
    keys["register"] = k_reg
    k_pub = sha256_canon([cache["register"][k_reg]["artifactDigest"], inputs.get("publishConfig")]) if k_reg and k_reg in cache.get("register", {}) else None
    keys["publish"] = k_pub
    return keys

def get_deps(node, inputs, cache, cacheKey, current_keys):
    d = {}
    if node == "verify_data":
        d["generation"], d["checksum"] = inputs.get("generation"), inputs.get("checksum")
    elif node == "prepare":
        d["canonicalData"], d["prepareCode"], d["prepareConfig"] = inputs.get("canonicalData"), inputs.get("prepareCode"), inputs.get("prepareConfig")
    elif node == "train":
        p_key = current_keys.get("prepare")
        d["prepareArtifact"] = cache["prepare"][p_key]["artifactDigest"] if p_key and p_key in cache.get("prepare", {}) else None
        d["trainCode"], d["trainConfig"], d["runtime"] = inputs.get("trainCode"), inputs.get("trainConfig"), inputs.get("runtime")
    elif node == "evaluate":
        t_key = current_keys.get("train")
        d["trainArtifact"] = cache["train"][t_key]["artifactDigest"] if t_key and t_key in cache.get("train", {}) else None
        d["canonicalData"], d["evaluateCode"], d["evaluateConfig"] = inputs.get("canonicalData"), inputs.get("evaluateCode"), inputs.get("evaluateConfig")
    elif node == "register":
        e_key = current_keys.get("evaluate")
        d["evaluateArtifact"] = cache["evaluate"][e_key]["artifactDigest"] if e_key and e_key in cache.get("evaluate", {}) else None
        d["schemaDigest"] = inputs.get("schemaDigest")
    elif node == "publish":
        r_key = current_keys.get("register")
        d["registerArtifact"] = cache["register"][r_key]["artifactDigest"] if r_key and r_key in cache.get("register", {}) else None
        d["publishConfig"] = inputs.get("publishConfig")
    d["cacheKey"] = cacheKey
    return d

# --- ENDPOINTS ---
@app.route('/', methods=['GET', 'POST'])
def health_check():
    return "Server is up and running!", 200

# -----------------------------------------
# NEW ENDPOINT: /verify-bundle
# -----------------------------------------
@app.route('/verify-bundle', methods=['POST'], strict_slashes=False)
def verify_bundle():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400

    policy = req.get("policy")
    files = req.get("files")
    if type(policy) is not dict or type(files) is not dict:
        return jsonify({"error": "INVALID_INPUT"}), 400

    violations = set()

    req_slices = policy.get("requiredSlices")
    license_p = policy.get("license")
    intent_p = policy.get("intendedUse")
    limits_p = policy.get("limitations")

    if (type(req_slices) is not list or not req_slices or not all(type(x) is str and x for x in req_slices) or
        type(license_p) is not str or not license_p or
        type(intent_p) is not str or not intent_p or
        type(limits_p) is not str or not limits_p):
        violations.add("INVALID_POLICY")

    required_files = [
        "README.md", "training_manifest.json", "evaluation.json",
        "inventory.json", "adapter_model.safetensors", "adapter_config.json"
    ]

    for rf in required_files:
        if rf not in files:
            violations.add(f"MISSING_FILE:{rf}")

    unsafe_exts = (".bin", ".pt", ".pth", ".pkl", ".pickle")
    for fn in files.keys():
        if fn.endswith(unsafe_exts):
            violations.add("UNSAFE_WEIGHTS")
        if fn not in required_files and fn != "inventory.json":
            violations.add("UNTRACKED_FILE")

    file_bytes_map = {}
    file_sha_map = {}
    for fn, fcontent in files.items():
        if type(fcontent) is not str:
            violations.add(f"INVALID_FILE:{fn}")
            continue
        b = fcontent.encode('utf-8')
        file_bytes_map[fn] = len(b)
        file_sha_map[fn] = hashlib.sha256(b).hexdigest()

    # Recompute inventory
    inv_entries = []
    for fn in sorted(files.keys()):
        if fn == "inventory.json": continue
        inv_entries.append({
            "name": fn,
            "bytes": file_bytes_map.get(fn, 0),
            "sha256": file_sha_map.get(fn, "")
        })
    inv_entries.sort(key=lambda x: x["name"].encode('utf-8'))
    recomputed_inv_json = json.dumps(inv_entries, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    recomputed_inv_digest = hashlib.sha256(recomputed_inv_json).hexdigest()

    if "inventory.json" in files:
        try:
            inv_data = json.loads(files["inventory.json"])
            if type(inv_data) is not list:
                violations.add("INVALID_JSON:inventory.json")
            else:
                for entry in inv_data:
                    if type(entry) is not dict or set(entry.keys()) != {"name", "bytes", "sha256"}:
                        violations.add("INVALID_JSON:inventory.json")
                        break
                expected_inv_arr = [{"name": e["name"], "bytes": e["bytes"], "sha256": e["sha256"]} for e in inv_entries]
                if inv_data != expected_inv_arr:
                    violations.add("INVENTORY_MISMATCH")
        except Exception:
            violations.add("INVALID_JSON:inventory.json")
    else:
        violations.add("INVALID_JSON:inventory.json")

    # adapter_config.json
    config_data = None
    if "adapter_config.json" in files:
        try:
            config_data = json.loads(files["adapter_config.json"])
            if type(config_data) is not dict:
                violations.add("INVALID_JSON:adapter_config.json")
                config_data = None
            else:
                r_val = config_data.get("r")
                tm_val = config_data.get("target_modules")
                if not (is_safe_pos_int(r_val) and type(tm_val) is list and tm_val and all(type(x) is str and x for x in tm_val) and len(set(tm_val)) == len(tm_val)):
                    violations.add("INVALID_ADAPTER_CONFIG")
        except Exception:
            violations.add("INVALID_JSON:adapter_config.json")

    # training_manifest.json
    manifest_data = None
    model_art_digest = None
    eval_art_digest = None
    base_rev = None

    if "adapter_model.safetensors" in files:
        model_art_digest = file_sha_map["adapter_model.safetensors"]
    if "evaluation.json" in files:
        eval_art_digest = file_sha_map["evaluation.json"]

    if "training_manifest.json" in files:
        try:
            manifest_data = json.loads(files["training_manifest.json"])
            if type(manifest_data) is not dict:
                violations.add("INVALID_JSON:training_manifest.json")
                manifest_data = None
            else:
                base_rev = manifest_data.get("baseRevision")
                if type(base_rev) is not str or not re.match(r'^[a-f0-9]{40}$', base_rev):
                    violations.add("MUTABLE_BASE_REVISION")

                mf_fields = ["task", "datasetDigest", "codeDigest", "trainingConfigDigest", "modelArtifactDigest", "evaluationArtifactDigest"]
                for mf in mf_fields:
                    val = manifest_data.get(mf)
                    if type(val) is not str or not val:
                        violations.add(f"MISSING_MANIFEST_FIELD:{mf}")

                if model_art_digest and manifest_data.get("modelArtifactDigest") != model_art_digest:
                    violations.add("MODEL_ARTIFACT_MISMATCH")
                if eval_art_digest and manifest_data.get("evaluationArtifactDigest") != eval_art_digest:
                    violations.add("EVALUATION_ARTIFACT_MISMATCH")
        except Exception:
            violations.add("INVALID_JSON:training_manifest.json")
    else:
        violations.add("INVALID_JSON:training_manifest.json")

    # evaluation.json
    eval_data = None
    if "evaluation.json" in files:
        try:
            eval_data = json.loads(files["evaluation.json"])
            if type(eval_data) is not dict:
                violations.add("INVALID_JSON:evaluation.json")
                eval_data = None
            else:
                if model_art_digest and eval_data.get("modelArtifactDigest") != model_art_digest:
                    violations.add("EVALUATION_DIGEST_MISMATCH")
                
                acc = eval_data.get("accuracy")
                if not is_finite_num(acc) or not (0 <= acc <= 1):
                    violations.add("INVALID_AGGREGATE")

                ev_slices = eval_data.get("slices")
                if type(ev_slices) is not dict:
                    if type(req_slices) is list:
                        for rs in req_slices:
                            violations.add(f"MISSING_SLICE:{rs}")
                else:
                    for rs in req_slices:
                        if rs not in ev_slices:
                            violations.add(f"MISSING_SLICE:{rs}")
                        else:
                            s_val = ev_slices[rs]
                            if not is_finite_num(s_val) or not (0 <= s_val <= 1):
                                violations.add(f"SLICE_RANGE:{rs}")
                            elif s_val < 0:
                                violations.add("INVALID_EVALUATION")
        except Exception:
            violations.add("INVALID_JSON:evaluation.json")

    # README.md (Model Card)
    readme_text = files.get("README.md", "")
    markers = list(re.finditer(r'<!--\s*tds-model-card\s*(\{.*?\})\s*-->', readme_text, re.DOTALL))
    
    if len(markers) == 0:
        violations.add("MISSING_MODEL_CARD")
    elif len(markers) > 1:
        violations.add("MODEL_CARD_COUNT")
    else:
        try:
            card_json_str = markers[0].group(1)
            card_data = json.loads(card_json_str)
            if type(card_data) is not dict:
                violations.add("INVALID_MODEL_CARD")
            else:
                if manifest_data:
                    if (card_data.get("task") != manifest_data.get("task") or
                        card_data.get("baseRevision") != manifest_data.get("baseRevision") or
                        card_data.get("datasetDigest") != manifest_data.get("datasetDigest") or
                        card_data.get("modelArtifactDigest") != manifest_data.get("modelArtifactDigest") or
                        card_data.get("license") != license_p or
                        card_data.get("intendedUse") != intent_p or
                        card_data.get("limitations") != limits_p):
                        violations.add("MODEL_CARD_MISMATCH")
        except Exception:
            violations.add("INVALID_MODEL_CARD")

    decision = "admit" if not violations else "reject"
    sorted_violations = sorted(list(violations), key=lambda x: x.encode('utf-8'))

    return jsonify({
        "decision": decision,
        "violations": sorted_violations,
        "inventoryDigest": recomputed_inv_digest
    })

# --- PREVIOUS ENDPOINTS (RETAINED) ---
@app.route('/pipeline', methods=['POST'], strict_slashes=False)
def pipeline():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_REQUEST"}), 409
    if not isinstance(req, dict): return jsonify({"error": "INVALID_REQUEST"}), 409
    sess, rev, inputs, events = req.get("session"), req.get("revision"), req.get("inputs"), req.get("events")
    if not isinstance(sess, str) or not sess or not is_safe_pos_int(rev) or not isinstance(inputs, dict): return jsonify({"error": "INVALID_REQUEST"}), 409
    for k in ["generation", "checksum", "canonicalData", "prepareCode", "prepareConfig", "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig", "schemaDigest", "publishConfig"]:
        if type(inputs.get(k)) is not str or not inputs[k]: return jsonify({"error": "INVALID_REQUEST"}), 409
    if not isinstance(events, list): return jsonify({"error": "INVALID_REQUEST"}), 409
    if sess not in pipeline_state:
        pipeline_state[sess] = {"session": sess, "revision": rev, "inputs": inputs, "cache": {n: {} for n in DAG_NODES}, "node_states": {}, "event_history": {}}
    else:
        state = pipeline_state[sess]
        if rev > state["revision"]:
            state["revision"], state["inputs"], state["node_states"] = rev, inputs, {}
        elif rev == state["revision"] and state["inputs"] != inputs: return jsonify({"error": "REVISION_CONFLICT"}), 409
    state = pipeline_state[sess]
    batch_node_states = {k: dict(v) for k, v in state["node_states"].items()}
    batch_cache = {n: {k: dict(v) for k, v in state["cache"][n].items()} for n in DAG_NODES}
    batch_history = dict(state["event_history"])
    accepted, ignored = [], []
    current_keys = compute_keys(state["inputs"], batch_cache)
    for e in events:
        if not (isinstance(e, dict) and set(e.keys()) == {"eventId", "revision", "node", "attempt", "status", "key", "artifactDigest", "receiptId"} and isinstance(e["eventId"], str) and is_safe_pos_int(e["revision"]) and isinstance(e["node"], str) and is_safe_pos_int(e["attempt"]) and e["status"] in ["started", "succeeded", "retryable_failed", "terminal_failed"] and isinstance(e["key"], str) and (e["artifactDigest"] is None or isinstance(e["artifactDigest"], str)) and (e["receiptId"] is None or isinstance(e["receiptId"], str))): return jsonify({"error": "INVALID_EVENT"}), 409
        c_json, eid = canon_json(e), e["eventId"]
        if eid in batch_history:
            if batch_history[eid] == c_json:
                if eid not in accepted: ignored.append(eid)
                continue
            else: return jsonify({"error": "EVENT_ID_CONFLICT"}), 409
        if e["revision"] != state["revision"] or e["node"] not in DAG_NODES or e["key"] != current_keys[e["node"]]: ignored.append(eid); continue
        if e["status"] == "succeeded":
            if not e["artifactDigest"]: ignored.append(eid); continue
            if e["node"] in ["register", "publish"]:
                if e["receiptId"] != f"receipt:{e['node']}:{e['key']}": ignored.append(eid); continue
            elif e["receiptId"] is not None: ignored.append(eid); continue
        else:
            if e["artifactDigest"] is not None or e["receiptId"] is not None: ignored.append(eid); continue
        node, key, cached, ns, action = e["node"], e["key"], batch_cache[e["node"]].get(e["key"]), batch_node_states.get(e["node"]), None
        if cached:
            if e["status"] == "succeeded":
                if e["artifactDigest"] != cached["artifactDigest"]: return jsonify({"error": "EVIDENCE_CONFLICT"}), 409
                else: action = "ignore"
            else: return jsonify({"error": "STATUS_CONFLICT"}), 409
        elif ns:
            if ns["status"] == "terminal_failed": return jsonify({"error": "STATUS_CONFLICT"}), 409
            elif ns["status"] == "started":
                if e["attempt"] < ns["attempt"]: action = "ignore"
                elif e["attempt"] == ns["attempt"]:
                    if e["status"] in ["succeeded", "retryable_failed", "terminal_failed"]: action = "accept"
                    else: return jsonify({"error": "STATUS_CONFLICT"}), 409
                else: return jsonify({"error": "STATUS_CONFLICT"}), 409
            elif ns["status"] == "retryable_failed":
                if e["attempt"] < ns["attempt"]: action = "ignore"
                elif e["attempt"] == ns["attempt"]: return jsonify({"error": "STATUS_CONFLICT"}), 409
                elif e["attempt"] == ns["attempt"] + 1 and e["status"] == "started": action = "accept"
                else: return jsonify({"error": "STATUS_CONFLICT"}), 409
        else:
            if e["attempt"] == 1 and e["status"] == "started": action = "accept"
            else: action = "ignore"
        if action == "accept":
            accepted.append(eid); batch_history[eid] = c_json
            if e["status"] == "succeeded":
                batch_cache[node][key] = {"artifactDigest": e["artifactDigest"], "eventId": eid, "receiptId": e["receiptId"]}
                if node in batch_node_states: del batch_node_states[node]
                current_keys = compute_keys(state["inputs"], batch_cache)
            else: batch_node_states[node] = {"status": e["status"], "attempt": e["attempt"], "eventId": eid, "key": key}
        else: ignored.append(eid)
    state["node_states"], state["cache"], state["event_history"] = batch_node_states, batch_cache, batch_history
    current_keys = compute_keys(state["inputs"], state["cache"])
    nodes_resp, ancestor_terminal = [], False
    for node in DAG_NODES:
        ckey = current_keys[node]
        cached = state["cache"][node].get(ckey) if ckey else None
        ns = state["node_states"].get(node)
        deps = get_deps(node, state["inputs"], state["cache"], ckey, current_keys)
        if cached: action_str, reason, trigger = "reuse", ["CACHE_HIT"], [cached["eventId"]]
        else:
            if ckey is not None:
                if ns:
                    if ns["status"] == "started": action_str, reason, trigger = "block", ["RUNNING"], [ns["eventId"]]
                    elif ns["status"] == "terminal_failed": action_str, reason, trigger, ancestor_terminal = "block", ["TERMINAL_FAILURE"], [ns["eventId"]], True
                    elif ns["status"] == "retryable_failed": action_str, reason, trigger = "rerun", ["RETRYABLE_FAILURE"], []
                else: action_str, reason, trigger = "rerun", ["CACHE_MISS"], []
            else: action_str, reason, trigger = "block", ["UPSTREAM_TERMINAL"] if ancestor_terminal else ["UPSTREAM_PENDING"], []
        nodes_resp.append({"node": node, "action": action_str, "reasonCodes": reason, "dependencyDigests": deps, "triggeringEventIds": trigger})
    return jsonify({"revision": state["revision"], "acceptedEventIds": accepted, "ignoredEventIds": ignored, "nodes": nodes_resp})

@app.route('/quantize', methods=['POST'], strict_slashes=False)
def quantize():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400
    phase = req.get("phase")
    if phase not in ["freeze", "select"]: return jsonify({"error": "INVALID_INPUT"}), 400
    if phase == "freeze":
        cands = req.get("candidates")
        if type(cands) is not list or len(cands) == 0: return jsonify({"error": "INVALID_INPUT"}), 400
        fid, cal_dig, tok_dig, aur = req.get("freezeId"), req.get("calibrationDigest"), req.get("tokenizerDigest"), req.get("allowedUnsupportedReasons")
        if fid in quantize_state:
            if quantize_state[fid]['request'] != req: return jsonify({"error": "FREEZE_ID_CONFLICT"}), 409
            return jsonify(quantize_state[fid]['response'])
        out_cands = []
        for c in cands:
            if type(c) is not dict: continue 
            cname = c.get("name")
            if type(cname) is not str: continue
            codes, files, inv, tb, pd = set(), c.get("files"), [], None, None
            if type(files) is not dict or not files: codes.add("INVALID_INPUT")
            else:
                valid_files = True
                for fn, fcontent in files.items():
                    if type(fn) is not str or not fn or type(fcontent) is not str: valid_files = False; codes.add("INVALID_INPUT"); break
                if valid_files:
                    tb = 0
                    for fn, fcontent in files.items():
                        b = fcontent.encode('utf-8')
                        inv.append({"name": fn, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
                    inv.sort(key=lambda x: x["name"].encode('utf-8'))
                    for x in inv: tb += x["bytes"]
                    pd = hashlib.sha256(json.dumps([{"name": x["name"], "bytes": x["bytes"], "sha256": x["sha256"]} for x in inv], separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
            status, ureason = "frozen", c.get("unsupportedReason")
            if "unsupportedReason" in c and type(ureason) is str:
                if type(aur) is list and ureason in aur: status = "unsupported"
                else: codes.add("UNALLOWED_UNSUPPORTED_REASON"); status = "invalid"
            else:
                if c.get("loadable") is not True: codes.add("NOT_LOADABLE")
                if c.get("calibrationDigest") != cal_dig: codes.add("CALIBRATION_MISMATCH")
                if c.get("tokenizerDigest") != tok_dig: codes.add("TOKENIZER_MISMATCH")
                if codes: status = "invalid"
            out_cands.append({"name": cname, "status": status, "inventory": inv if "INVALID_INPUT" not in codes else [], "totalBytes": tb, "packageDigest": pd, "reasonCodes": sorted(list(codes), key=lambda x: x.encode('utf-8'))})
        out_cands.sort(key=lambda x: x["name"].encode('utf-8'))
        res = {"freezeId": fid, "candidates": out_cands}
        if type(fid) is str and fid: quantize_state[fid] = {"request": req, "response": res}
        return jsonify(res)
    elif phase == "select":
        cands, rows, policy, fid, lats = req.get("candidates"), req.get("rows"), req.get("policy"), req.get("freezeId"), req.get("latencies")
        if type(cands) is not list or type(rows) is not list or type(policy) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400
        is_policy_valid = True
        mb, af, rs, ml, co = policy.get("maxBytes"), policy.get("aggregateFloor"), policy.get("requiredSlices"), policy.get("maxLatencyMs"), policy.get("candidateOrder")
        if not (is_safe_pos_int(mb) or mb==0 and is_finite_num(af) and 0 <= af <= 1 and type(rs) is dict and is_finite_num(ml) and ml >= 0 and type(co) is list): is_policy_valid = False
        else:
            for k, v in rs.items():
                if type(k) is not str or not is_finite_num(v) or not (0 <= v <= 1): is_policy_valid = False; break
            for x in co:
                if type(x) is not str: is_policy_valid = False; break
            if len(set(co)) != len(co): is_policy_valid = False
        cand_names = [c.get("name") for c in cands if type(c) is dict and type(c.get("name")) is str]
        if is_policy_valid and set(cand_names) != set(co): is_policy_valid = False
        is_frozen = type(fid) is str and fid in quantize_state
        stored_cands = quantize_state[fid]["response"]["candidates"] if is_frozen else []
        stored_cands_map = {c["name"]: c for c in stored_cands}
        global_lineage_ok = True
        if is_frozen and cands != stored_cands: global_lineage_ok = False
        results, admitted_list = [], []
        for c in cands:
            if type(c) is not dict: continue
            cname = c.get("name")
            if type(cname) is not str: continue
            codes = set()
            if not is_frozen: codes.add("NOT_FROZEN")
            if not is_policy_valid: codes.add("INVALID_POLICY")
            if is_frozen and (not global_lineage_ok or cname not in stored_cands_map or c != stored_cands_map[cname]): codes.add("INVALID_LINEAGE")
            cinv = c.get("inventory")
            if type(cinv) is list:
                cb = 0
                for x in cinv:
                    if type(x) is dict and type(x.get("bytes")) is int: cb += x["bytes"]
                cpd = hashlib.sha256(json.dumps([{"name": x.get("name"), "bytes": x.get("bytes"), "sha256": x.get("sha256")} for x in cinv if type(x) is dict], separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
                if c.get("totalBytes") != cb or c.get("packageDigest") != cpd: codes.add("INVALID_MANIFEST")
            else: codes.add("INVALID_MANIFEST")
            lat = lats.get(cname) if type(lats) is dict else None
            valid_lat = is_finite_num(lat) and lat >= 0
            preds_valid = True
            for r in rows:
                if type(r) is not dict or "predictions" not in r or type(r["predictions"]) is not dict: preds_valid = False; break
                if r["predictions"].get(cname) not in [0, 1] or r.get("label") not in [0, 1]: preds_valid = False; break
            agg, sl_acc = None, None
            if not preds_valid: codes.add("INVALID_PREDICTIONS")
            else:
                sl_acc = {}
                if rows:
                    agg = round(sum(1 for r in rows if r.get("label") == r["predictions"].get(cname)) / len(rows), 12)
                    smets = {}
                    for r in rows:
                        sl = r.get("slice")
                        if type(sl) is str:
                            if sl not in smets: smets[sl] = {"c": 0, "t": 0}
                            smets[sl]["t"] += 1
                            if r.get("label") == r["predictions"].get(cname): smets[sl]["c"] += 1
                    for sl, md in smets.items(): sl_acc[sl] = round(md["c"] / md["t"], 12)
                else: agg = 0.0
            if is_policy_valid:
                if not valid_lat or lat > ml: codes.add("LATENCY_LIMIT")
                tb = c.get("totalBytes")
                if not is_safe_pos_int(tb) and tb != 0 or tb > mb: codes.add("SIZE_LIMIT")
                if preds_valid:
                    if agg < af: codes.add("AGGREGATE_FLOOR")
                    for req_s, floor in rs.items():
                        if req_s not in sl_acc: codes.add(f"MISSING_SLICE:{req_s}")
                        elif sl_acc[req_s] < floor: codes.add(f"SLICE_FLOOR:{req_s}")
            is_admitted = not codes and is_frozen and stored_cands_map.get(cname, {}).get("status") == "frozen"
            valid_tb = (is_safe_pos_int(c.get("totalBytes")) or c.get("totalBytes") == 0) and "INVALID_MANIFEST" not in codes
            res_obj = {"name": cname, "aggregate": agg, "slices": sl_acc, "totalBytes": c.get("totalBytes") if valid_tb else None, "latencyMs": lat if valid_lat else None, "admitted": is_admitted, "reasonCodes": sorted(list(codes), key=lambda x: x.encode('utf-8'))}
            results.append(res_obj)
            if is_admitted: admitted_list.append(res_obj)
        def get_order_idx(n):
            try: return co.index(n) if type(co) is list else 999999
            except: return 999999
        results.sort(key=lambda x: (get_order_idx(x["name"]), x["name"].encode('utf-8')))
        admitted_list.sort(key=lambda x: (x["totalBytes"], x["latencyMs"], get_order_idx(x["name"])))
        selected = admitted_list[0]["name"] if admitted_list else None
        return jsonify({"freezeId": fid, "selected": selected, "results": results, "packageManifest": stored_cands_map.get(selected) if selected and is_frozen else None})

@app.route('/adapt', methods=['POST'], strict_slashes=False)
def adapt():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400
    op = req.get("operation")
    if op not in ["choose", "repair"]: return jsonify({"error": "INVALID_INPUT"}), 400
    if op == "choose":
        policy, cands = req.get("policy"), req.get("candidates")
        if type(policy) is not dict or type(cands) is not list: return jsonify({"error": "INVALID_INPUT"}), 400
        req_names = ["prompt_only", "retrieval", "lora", "qlora"]
        cand_map = {}
        for c in cands:
            if type(c) is dict and "name" in c and type(c["name"]) is str: cand_map[c["name"]] = cand_map.get(c["name"], 0) + 1
        for n in req_names:
            if cand_map.get(n, 0) != 1: return jsonify({"error": "INVALID_INPUT"}), 400
        p_mq, p_fr, p_ml, p_mm, p_le, p_mt, p_hr = policy.get("minQuality"), policy.get("freshnessRequired"), policy.get("maxLatencyMs"), policy.get("maxMemoryMb"), policy.get("maxLabeledExamples"), policy.get("maxTotalCost"), policy.get("horizonRequests")
        if not (is_finite_num(p_mq) and 0 <= p_mq <= 1 and type(p_fr) is bool and is_finite_num(p_ml) and p_ml >= 0 and is_finite_num(p_mm) and p_mm >= 0 and is_safe_int(p_le) and is_finite_num(p_mt) and p_mt >= 0 and is_safe_int(p_hr)): return jsonify({"error": "INVALID_INPUT"}), 400
        total_costs, reason_codes, eligible = {}, {n: set() for n in req_names}, []
        cand_dicts = {c["name"]: c for c in cands if type(c) is dict and c.get("name") in req_names}
        for n in req_names:
            c = cand_dicts[n]
            avail, qual, fresh, lat, mem, lab, otc, rc = c.get("available"), c.get("quality"), c.get("freshness"), c.get("latencyMs"), c.get("memoryMb"), c.get("labeledExamples"), c.get("oneTimeCost"), c.get("recurringCost")
            if not (type(avail) is bool and is_finite_num(qual) and 0 <= qual <= 1 and type(fresh) is bool and is_finite_num(lat) and lat >= 0 and is_finite_num(mem) and mem >= 0 and is_safe_int(lab) and is_finite_num(otc) and otc >= 0 and is_finite_num(rc) and rc >= 0):
                reason_codes[n].add("INVALID_INPUT"); total_costs[n] = 0.0; continue
            cost = round(otc + p_hr * rc, 12)
            total_costs[n] = cost
            if not avail: reason_codes[n].add("UNAVAILABLE")
            if qual < p_mq: reason_codes[n].add("QUALITY_FLOOR")
            if p_fr and not fresh: reason_codes[n].add("FRESHNESS_REQUIRED")
            if lat > p_ml: reason_codes[n].add("LATENCY_LIMIT")
            if mem > p_mm: reason_codes[n].add("MEMORY_LIMIT")
            if lab > p_le: reason_codes[n].add("DATA_LIMIT")
            if cost > p_mt: reason_codes[n].add("COST_LIMIT")
            if not reason_codes[n]: eligible.append(n)
        return jsonify({"selected": eligible[0] if eligible else None, "eligible": eligible, "totalCosts": total_costs, "reasonCodes": {n: sorted(list(reason_codes[n]), key=lambda x: x.encode('utf-8')) for n in req_names}})
    if op == "repair":
        reason_codes, toks, tok_valid, labels = set(), req.get("tokens"), True, []
        if type(toks) is not list or not toks: tok_valid = False; reason_codes.add("INVALID_TOKEN")
        else:
            for t in toks:
                if type(t) is not dict or not is_safe_int(t.get("id")) or t.get("role") not in ["system", "user", "assistant"] or type(t.get("padding")) is not bool or type(t.get("text")) is not str: tok_valid = False; break
            if not tok_valid: reason_codes.add("INVALID_TOKEN"); labels = [-100] * len(toks) if type(toks) is list else []
            else:
                for t in toks: labels.append(t["id"] if t["role"] == "assistant" and t["padding"] is False else -100)
        t_pass = True
        if req.get("templateApplications") != 1: t_pass = False; reason_codes.add("CHAT_TEMPLATE_COUNT")
        params, allowed_t, peft_pass, trainable, trainable_count = req.get("parameters"), req.get("allowedTargets"), True, [], 0
        if type(params) is not list or type(allowed_t) is not list or not allowed_t: peft_pass = False; reason_codes.add("INVALID_PARAMETER")
        else:
            tgt_set, tgt_valid = set(), True
            for at in allowed_t:
                if type(at) is not str or not at or at in tgt_set: tgt_valid = False; break
                tgt_set.add(at)
            p_names, p_valid = set(), True
            for p in params:
                if type(p) is not dict or type(p.get("name")) is not str or type(p.get("target")) is not str or not is_safe_pos_int(p.get("numel")) or p.get("name") in p_names: p_valid = False; break
                p_names.add(p.get("name"))
            if not tgt_valid or not p_valid: peft_pass = False; reason_codes.add("INVALID_PARAMETER")
            else:
                for p in params:
                    if p.get("target") in tgt_set and (p.get("name").endswith(".lora_A.weight") or p.get("name").endswith(".lora_B.weight")):
                        trainable.append(p.get("name")); trainable_count += p.get("numel")
                if not trainable: peft_pass = False; reason_codes.add("INVALID_PARAMETER")
                else: trainable.sort(key=lambda x: x.encode('utf-8'))
        if req.get("inferenceMode") is not False: reason_codes.add("INFERENCE_MODE")
        if req.get("dropoutActiveDuringEval") is not False: reason_codes.add("EVAL_DROPOUT_ACTIVE")
        t_rows, e_rows = req.get("trainRowIds"), req.get("evalRowIds")
        if type(t_rows) is not list or type(e_rows) is not list or not t_rows or not e_rows: reason_codes.add("EVAL_LEAKAGE")
        else:
            t_set, t_ok, e_set, e_ok = set(), True, set(), True
            for r in t_rows:
                if type(r) is not str or not r or r in t_set: t_ok = False; break
                t_set.add(r)
            for r in e_rows:
                if type(r) is not str or not r or r in e_set: e_ok = False; break
                e_set.add(r)
            if not t_ok or not e_ok or not t_set.isdisjoint(e_set): reason_codes.add("EVAL_LEAKAGE")
        af, adapter_files = req.get("artifactFiles"), []
        if type(af) is not list: reason_codes.add("ADAPTER_FILE_SET")
        else:
            if "pytorch_model.bin" in af or "model.safetensors" in af: reason_codes.add("FULL_MODEL_ARTIFACT")
            elif set(af) != {"adapter_config.json", "adapter_model.safetensors"} or len(af) != 2: reason_codes.add("ADAPTER_FILE_SET")
            else: adapter_files = sorted(af, key=lambda x: x.encode('utf-8'))
        br, dd, cd, cfd, ed = req.get("baseRevision"), req.get("datasetDigest"), req.get("codeDigest"), req.get("configDigest"), req.get("expectedDigests")
        if type(br) is not str or not re.match(r'^[a-f0-9]{40}$', br): reason_codes.add("MUTABLE_BASE_REVISION")
        if (type(ed) is not dict or type(dd) is not str or not re.match(r'^[a-f0-9]{64}$', dd) or dd != ed.get("dataset") or type(cd) is not str or not re.match(r'^[a-f0-9]{64}$', cd) or cd != ed.get("code") or type(cfd) is not str or not re.match(r'^[a-f0-9]{64}$', cfd) or cfd != ed.get("config")): reason_codes.add("LINEAGE_MISMATCH")
        mb, ga, rep, eeb = req.get("microBatch"), req.get("gradientAccumulation"), req.get("replicas"), req.get("expectedEffectiveBatch")
        if not (is_safe_pos_int(mb) and is_safe_pos_int(ga) and is_safe_pos_int(rep) and is_safe_pos_int(eeb) and mb * ga * rep == eeb): reason_codes.add("EFFECTIVE_BATCH_MISMATCH")
        chk = req.get("checkpoint")
        if type(chk) is not dict or set(chk.keys()) != {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}: reason_codes.add("INCOMPLETE_CHECKPOINT")
        uw, rw, rt = req.get("uninterruptedWeights"), req.get("resumedWeights"), req.get("resumeTolerance")
        if (type(uw) is not list or not uw or type(rw) is not list or not rw or len(uw) != len(rw) or not is_finite_num(rt) or rt < 0): reason_codes.add("RESUME_DIVERGENCE")
        else:
            w_ok = True
            for u, r in zip(uw, rw):
                if not is_finite_num(u) or not is_finite_num(r) or abs(u - r) > rt: w_ok = False; break
            if not w_ok: reason_codes.add("RESUME_DIVERGENCE")
        return jsonify({"labels": labels, "templatePass": t_pass, "trainableParams": trainable, "trainableCount": trainable_count, "peftConfigPass": peft_pass, "adapterFiles": adapter_files, "checkpointComplete": "INCOMPLETE_CHECKPOINT" not in reason_codes, "lineagePass": "MUTABLE_BASE_REVISION" not in reason_codes and "LINEAGE_MISMATCH" not in reason_codes, "evalIsolated": "EVAL_LEAKAGE" not in reason_codes, "evaluationDeterministic": "INFERENCE_MODE" not in reason_codes and "EVAL_DROPOUT_ACTIVE" not in reason_codes, "resumePass": "RESUME_DIVERGENCE" not in reason_codes, "reasonCodes": sorted(list(reason_codes), key=lambda x: x.encode('utf-8'))})

@app.route('/promote', methods=['POST'], strict_slashes=False)
def promote():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400
    policy, versions, champ_ver = req.get("policy"), req.get("versions"), req.get("championVersion")
    if type(policy) is not dict or type(versions) is not list or type(champ_ver) is not str: return jsonify({"error": "INVALID_INPUT"}), 400
    dt_asof = parse_iso8601(req.get("asOf"))
    is_policy_valid = True
    if type(policy.get("datasetDigest")) is not str or not policy.get("datasetDigest"): is_policy_valid = False
    if type(policy.get("schemaDigest")) is not str or not policy.get("schemaDigest"): is_policy_valid = False
    if type(policy.get("maxAgeSeconds")) is not int or isinstance(policy.get("maxAgeSeconds"), bool) or policy.get("maxAgeSeconds") < 0: is_policy_valid = False
    if type(policy.get("maxSizeBytes")) is not int or isinstance(policy.get("maxSizeBytes"), bool) or policy.get("maxSizeBytes") < 0: is_policy_valid = False
    acc_floor = policy.get("accuracyFloor")
    if type(acc_floor) not in (int, float) or isinstance(acc_floor, bool) or not math.isfinite(acc_floor) or not (0 <= acc_floor <= 1): is_policy_valid = False
    lat_max = policy.get("maxLatencyMs")
    if type(lat_max) not in (int, float) or isinstance(lat_max, bool) or not math.isfinite(lat_max) or lat_max < 0: is_policy_valid = False
    min_imp = policy.get("minImprovement")
    if type(min_imp) not in (int, float) or isinstance(min_imp, bool) or not math.isfinite(min_imp) or not (0 <= min_imp <= 1): is_policy_valid = False
    req_slices = policy.get("requiredSlices")
    if type(req_slices) is not dict: is_policy_valid = False
    else:
        for k, v in req_slices.items():
            if type(k) is not str or type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) or not (0 <= v <= 1): is_policy_valid = False
    v_counts = {}
    for v in versions:
        if type(v) is dict and type(v.get("version")) is str: v_counts[v["version"]] = v_counts.get(v["version"], 0) + 1
    failed_gates, eligible = {}, []
    for v in versions:
        if type(v) is not dict: continue
        vid = v.get("version")
        vid_str = str(vid) if type(vid) is not str else vid
        if vid_str not in failed_gates: failed_gates[vid_str] = set()
        codes = failed_gates[vid_str]
        is_canonical = type(vid) is str and bool(re.match(r'^[1-9]\d*$', vid)) and int(vid) <= 9007199254740991
        if not is_canonical: codes.add("INVALID_VERSION")
        if type(vid) is str and v_counts.get(vid, 0) > 1: codes.add("DUPLICATE_VERSION")
        if not is_policy_valid: codes.add("INVALID_POLICY")
        eval_obj = v.get("evaluation")
        if type(eval_obj) is not dict: codes.add("MISSING_EVALUATION")
        else:
            c_at = eval_obj.get("createdAt")
            dt_cat = parse_iso8601(c_at)
            if not dt_cat or not dt_asof: codes.add("INVALID_TIMESTAMP")
            else:
                delta = (dt_asof - dt_cat).total_seconds()
                if delta < 0: codes.add("FUTURE_EVALUATION")
                elif is_policy_valid and delta > policy.get("maxAgeSeconds", 0): codes.add("STALE_EVALUATION")
            if eval_obj.get("artifactDigest") != v.get("artifactDigest"): codes.add("ARTIFACT_MISMATCH")
            if is_policy_valid:
                if eval_obj.get("datasetDigest") != policy.get("datasetDigest"): codes.add("DATASET_MISMATCH")
                if eval_obj.get("schemaDigest") != policy.get("schemaDigest"): codes.add("SCHEMA_MISMATCH")
            acc, lat, sz = eval_obj.get("accuracy"), eval_obj.get("latencyMs"), eval_obj.get("sizeBytes")
            if not is_finite_num(acc) or not is_finite_num(lat) or not is_finite_num(sz): codes.add("NON_FINITE")
            else:
                if not (0 <= acc <= 1) or lat < 0 or sz < 0 or type(sz) is not int or isinstance(sz, bool): codes.add("METRIC_RANGE")
                elif is_policy_valid:
                    if acc < policy.get("accuracyFloor", 0): codes.add("ACCURACY_FLOOR")
                    if lat > policy.get("maxLatencyMs", 0): codes.add("LATENCY_LIMIT")
                    if sz > policy.get("maxSizeBytes", 0): codes.add("SIZE_LIMIT")
            eval_slices = eval_obj.get("slices")
            if is_policy_valid:
                if type(eval_slices) is not dict:
                    for req_s in req_slices.keys(): codes.add(f"MISSING_SLICE:{req_s}")
                else:
                    for req_s, floor in req_slices.items():
                        if req_s not in eval_slices: codes.add(f"MISSING_SLICE:{req_s}")
                        else:
                            s_val = eval_slices[req_s]
                            if not is_finite_num(s_val) or not (0 <= s_val <= 1): codes.add(f"SLICE_RANGE:{req_s}")
                            elif s_val < floor: codes.add(f"SLICE_FLOOR:{req_s}")
        if not codes and is_canonical: eligible.append(v)
    eligible.sort(key=lambda x: (-x['evaluation']['accuracy'], x['evaluation']['latencyMs'], x['evaluation']['sizeBytes'], int(x['version'])))
    champ_v = next((v for v in eligible if v['version'] == champ_ver), None)
    if not champ_v: action, sel_ver, evidence, alias_mut = "block", None, None, None
    else:
        challenger = eligible[0]
        if challenger['version'] == champ_ver: action, sel_ver, alias_mut, evidence = "retain", champ_ver, None, champ_v['evaluation']
        else:
            diff = round(challenger['evaluation']['accuracy'] - champ_v['evaluation']['accuracy'], 12)
            if diff >= min_imp: action, sel_ver, alias_mut, evidence = "promote", challenger['version'], {"alias": "champion", "version": challenger['version']}, challenger['evaluation']
            else: action, sel_ver, alias_mut, evidence = "retain", champ_ver, None, champ_v['evaluation']
    return jsonify({"action": action, "championVersion": champ_ver, "selectedVersion": sel_ver, "eligibleVersions": [v['version'] for v in eligible], "failedGates": {k: sorted(list(v_set), key=lambda x: x.encode('utf-8')) for k, v_set in failed_gates.items()}, "aliasMutation": alias_mut, "evidence": evidence})

@app.route('/bqml', methods=['POST'], strict_slashes=False)
def bqml():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict or req.get('phase') not in ['select', 'evaluate']: return jsonify({"error": "INVALID_INPUT"}), 400
    phase = req['phase']
    if phase == 'select':
        runId = req.get('runId')
        if type(runId) is not str or not runId or len(runId) > 128: return jsonify({"runId": runId if type(runId) is str else None, "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": ["INVALID_INPUT"]})
        if runId in bqml_state:
            if bqml_state[runId]['request'] != req: return jsonify({"error": "RUN_ID_CONFLICT"}), 409
            return jsonify(bqml_state[runId]['response'])
        reasonCodes = set()
        fb_features, limit, rows, trials = req.get('forbiddenFeatures'), req.get('numTrialsLimit'), req.get('rows'), req.get('trials')
        if (type(fb_features) is not list or type(limit) is not int or limit <= 0 or type(limit) is bool or type(rows) is not list or not rows or type(trials) is not list): reasonCodes.add("INVALID_INPUT")
        valid_rows = []
        if "INVALID_INPUT" not in reasonCodes:
            row_ids = set()
            for r in rows:
                if type(r) is not dict: reasonCodes.add("INVALID_INPUT"); break
                rid, ent, et = r.get('id'), r.get('entity'), r.get('eventTime')
                pt, ver, sp = r.get('predictionTime'), r.get('version'), r.get('split')
                fts = r.get('features')
                if (type(rid) is not str or type(ent) is not str or type(et) is not str or type(pt) is not str or type(sp) is not str or type(fts) is not dict): reasonCodes.add("INVALID_INPUT"); break
                if type(ver) is not int or type(ver) is bool or ver < 0 or ver > 9007199254740991: reasonCodes.add("INVALID_INPUT"); break
                if sp not in ["TRAIN", "EVAL"]: reasonCodes.add("INVALID_INPUT"); break
                dt_et, dt_pt = parse_iso8601(et), parse_iso8601(pt)
                if not dt_et or not dt_pt: reasonCodes.add("INVALID_INPUT"); break
                if rid in row_ids: reasonCodes.add("INVALID_INPUT"); break
                row_ids.add(rid)
                for fn, fv in fts.items():
                    if type(fv) is not dict or 'availableAt' not in fv or not parse_iso8601(fv['availableAt']): reasonCodes.add("INVALID_INPUT"); break
                if "INVALID_INPUT" in reasonCodes: break
                r_copy = dict(r)
                r_copy['_dt_et'] = dt_et.astimezone(timezone.utc)
                r_copy['_dt_pt'] = dt_pt.astimezone(timezone.utc)
                valid_rows.append(r_copy)
        trial_ids = set()
        if "INVALID_INPUT" not in reasonCodes:
            for t in trials:
                if type(t) is not dict: reasonCodes.add("INVALID_INPUT"); break
                tid, st, em = t.get('trialId'), t.get('status'), t.get('evalMetric')
                if type(tid) is not int or type(tid) is bool or tid < 0 or tid > 9007199254740991 or st not in ["SUCCEEDED", "FAILED"] or not is_finite_num(em): reasonCodes.add("INVALID_INPUT"); break
                if tid in trial_ids: reasonCodes.add("INVALID_INPUT"); break
                trial_ids.add(tid)
            if len(trials) > limit and "INVALID_INPUT" not in reasonCodes: reasonCodes.add("TRIAL_LIMIT_EXCEEDED")
        train_ids, eval_ids, final_features, best_trial = [], [], [], None
        if not reasonCodes:
            dedup = {}
            for r in valid_rows:
                k = (r['entity'], r['_dt_et'].strftime('%Y-%m-%dT%H:%M:%S.%f')[:23] + 'Z')
                if k not in dedup: dedup[k] = r
                else:
                    exist = dedup[k]
                    if r['version'] > exist['version'] or (r['version'] == exist['version'] and r['id'].encode('utf-8') < exist['id'].encode('utf-8')): dedup[k] = r
            final_rows = list(dedup.values())
            all_fs = {}
            for r in final_rows:
                for fn, fv in r['features'].items():
                    if fn not in all_fs: all_fs[fn] = []
                    all_fs[fn].append({'pt': r['_dt_pt'], 'at': parse_iso8601(fv['availableAt']).astimezone(timezone.utc)})
            for fn, times in all_fs.items():
                if fn in fb_features: continue
                if len(times) != len(final_rows): continue
                if all(t['at'] <= t['pt'] for t in times): final_features.append(fn)
            train_ids = sorted([r['id'] for r in final_rows if r['split'] == "TRAIN"], key=lambda x: x.encode('utf-8'))
            eval_ids = sorted([r['id'] for r in final_rows if r['split'] == "EVAL"], key=lambda x: x.encode('utf-8'))
            final_features.sort(key=lambda x: x.encode('utf-8'))
            succ = [t for t in trials if t['status'] == "SUCCEEDED"]
            if not succ: reasonCodes.add("NO_SUCCESSFUL_TRIAL")
            else:
                succ.sort(key=lambda x: (-x['evalMetric'], x['trialId']))
                best_trial = succ[0]['trialId']
        if reasonCodes: res = {"runId": runId, "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [], "featureNames": [], "datasetDigest": None, "reasonCodes": sorted(list(reasonCodes), key=lambda x: x.encode('utf-8'))}
        else:
            digest_str = json.dumps({"trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": final_features}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
            res = {"runId": runId, "selectedTrialId": best_trial, "trainRowIds": train_ids, "evalRowIds": eval_ids, "featureNames": final_features, "datasetDigest": hashlib.sha256(digest_str).hexdigest(), "reasonCodes": []}
        bqml_state[runId] = {'request': req, 'response': res}
        return jsonify(res)
    elif phase == 'evaluate':
        runId, s_trial, d_digest = req.get('runId'), req.get('selectedTrialId'), req.get('datasetDigest')
        m_floor, r_slices, rows = req.get('metricFloor'), req.get('requiredSlices'), req.get('rows')
        b_proc, m_bytes = req.get('bytesProcessed'), req.get('maxBytes')
        reasonCodes, c_pass = set(), True
        if (type(runId) is not str or type(s_trial) is not int or type(s_trial) is bool or type(d_digest) is not str or not is_finite_num(m_floor) or not (0 <= m_floor <= 1) or type(r_slices) is not dict or type(rows) is not list or type(b_proc) is not int or type(b_proc) is bool or b_proc < 0 or type(m_bytes) is not int or type(m_bytes) is bool or m_bytes < 0): reasonCodes.add("INVALID_INPUT"); c_pass = False
        if "INVALID_INPUT" not in reasonCodes:
            for k, v in r_slices.items():
                if type(k) is not str or not is_finite_num(v) or not (0 <= v <= 1): reasonCodes.add("INVALID_INPUT"); c_pass = False; break
        lineage_ok = False
        if "INVALID_INPUT" not in reasonCodes:
            if runId in bqml_state and bqml_state[runId]['response'].get('selectedTrialId') == s_trial and bqml_state[runId]['response'].get('datasetDigest') == d_digest and s_trial is not None: lineage_ok = True
            if not lineage_ok: reasonCodes.add("INVALID_LINEAGE"); c_pass = False
        test_metric, rows_ok = None, True
        if "INVALID_INPUT" not in reasonCodes:
            if not rows: rows_ok = False
            else:
                for r in rows:
                    if type(r) is not dict: rows_ok = False; break
                    if r.get('label') not in [0, 1] or r.get('prediction') not in [0, 1] or type(r.get('slice')) is not str or not r.get('slice'): rows_ok = False; break
            if not rows_ok: reasonCodes.add("INVALID_TEST_ROW"); c_pass = False
            if lineage_ok and rows_ok:
                test_metric = round(sum(1 for r in rows if r['label'] == r['prediction']) / len(rows), 12) if len(rows) > 0 else 0.0
                if test_metric < m_floor: reasonCodes.add("AGGREGATE_FLOOR")
                slice_mets = {}
                for r in rows:
                    sl = r['slice']
                    if sl not in slice_mets: slice_mets[sl] = {'c': 0, 't': 0}
                    slice_mets[sl]['t'] += 1
                    if r['label'] == r['prediction']: slice_mets[sl]['c'] += 1
                for req_s, floor in r_slices.items():
                    if req_s not in slice_mets: reasonCodes.add(f"MISSING_SLICE:{req_s}"); c_pass = False
                    elif round(slice_mets[req_s]['c'] / slice_mets[req_s]['t'], 12) < floor: reasonCodes.add(f"SLICE_FLOOR:{req_s}"); c_pass = False
        if "INVALID_INPUT" not in reasonCodes and b_proc > m_bytes: reasonCodes.add("BYTE_LIMIT")
        return jsonify({"runId": runId if type(runId) is str else None, "selectedTrialId": s_trial if type(s_trial) is int and not isinstance(s_trial, bool) else None, "datasetDigest": d_digest if type(d_digest) is str else None, "testMetric": test_metric, "criticalSlicePass": c_pass, "decision": "reject" if reasonCodes else "admit", "bytesProcessed": b_proc if type(b_proc) is int and not isinstance(b_proc, bool) else None, "reasonCodes": sorted(list(reasonCodes), key=lambda x: x.encode('utf-8'))})

@app.route('/build-corpus', methods=['POST'], strict_slashes=False)
def build_corpus():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400
    if type(req) is not dict or type(req.get('policy')) is not dict or type(req.get('objects')) is not list: return jsonify({"error": "INVALID_INPUT"}), 400
    policy = req['policy']
    min_dt, max_dt, thresh = parse_iso8601(policy.get('minTime')), parse_iso8601(policy.get('maxTime')), policy.get('contaminationThreshold')
    policy_valid = bool(min_dt and max_dt and type(thresh) in (int, float) and not isinstance(thresh, bool) and 0 <= thresh <= 1)
    rej_objs, rej_rows, lineage, valid_rows = [], [], [], []
    for item in req['objects']:
        obj, obj_codes = item if type(item) is dict else {}, set()
        uri = obj.get('uri')
        if type(uri) is not str or uri != "gs://bucket/object": obj_codes.add("URI_INVALID")
        gen, f_gen = obj.get('generation'), obj.get('fetchedGeneration')
        if not (type(gen) is str and gen.isdecimal()) or not (type(f_gen) is str and f_gen.isdecimal()): obj_codes.add("GENERATION_INVALID")
        if gen != f_gen: obj_codes.add("GENERATION_MISMATCH")
        crc_in = obj.get('crc32c')
        crc_syntax_valid = type(crc_in) is str and bool(re.match(r'^[a-f0-9]{8}$', crc_in))
        if not crc_syntax_valid: obj_codes.add("CRC32C_INVALID")
        content = obj.get('content')
        if type(content) is not str: obj_codes.add("SCHEMA_INVALID")
        elif crc_syntax_valid and google_crc32c.Checksum(content.encode('utf-8')).hexdigest().decode('utf-8').lower().zfill(8) != crc_in: obj_codes.add("CRC32C_MISMATCH")
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
