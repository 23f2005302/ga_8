import json
import re
import unicodedata
import hashlib
import math
from datetime import datetime, timezone
import google_crc32c
from flask import Flask, request, jsonify

app = Flask(__name__)
bqml_state = {}
quantize_state = {}
pipeline_state = {}

DAG_NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]

def is_safe_pos_int(x):
    return type(x) is int and not isinstance(x, bool) and 1 <= x <= 9007199254740991

def canon_json(data):
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)

def sha256_canon(data):
    return hashlib.sha256(canon_json(data).encode('utf-8')).hexdigest()

def compute_keys(inputs, cache):
    keys = {}
    k_vd = sha256_canon([inputs.get("generation"), inputs.get("checksum")])
    keys["verify_data"] = k_vd

    k_prep = None
    if k_vd and k_vd in cache.get("verify_data", {}):
        k_prep = sha256_canon([inputs.get("canonicalData"), inputs.get("prepareCode"), inputs.get("prepareConfig")])
    keys["prepare"] = k_prep

    k_train = None
    if k_prep and k_prep in cache.get("prepare", {}):
        k_train = sha256_canon([cache["prepare"][k_prep]["artifactDigest"], inputs.get("trainCode"), inputs.get("trainConfig"), inputs.get("runtime")])
    keys["train"] = k_train

    k_eval = None
    if k_train and k_train in cache.get("train", {}):
        k_eval = sha256_canon([cache["train"][k_train]["artifactDigest"], inputs.get("canonicalData"), inputs.get("evaluateCode"), inputs.get("evaluateConfig")])
    keys["evaluate"] = k_eval

    k_reg = None
    if k_eval and k_eval in cache.get("evaluate", {}):
        k_reg = sha256_canon([cache["evaluate"][k_eval]["artifactDigest"], inputs.get("schemaDigest")])
    keys["register"] = k_reg

    k_pub = None
    if k_reg and k_reg in cache.get("register", {}):
        k_pub = sha256_canon([cache["register"][k_reg]["artifactDigest"], inputs.get("publishConfig")])
    keys["publish"] = k_pub

    return keys

def get_deps(node, inputs, cache, cacheKey, current_keys):
    d = {}
    if node == "verify_data":
        d["generation"] = inputs.get("generation")
        d["checksum"] = inputs.get("checksum")
    elif node == "prepare":
        d["canonicalData"] = inputs.get("canonicalData")
        d["prepareCode"] = inputs.get("prepareCode")
        d["prepareConfig"] = inputs.get("prepareConfig")
    elif node == "train":
        p_key = current_keys.get("prepare")
        d["prepareArtifact"] = cache["prepare"][p_key]["artifactDigest"] if p_key and p_key in cache.get("prepare", {}) else None
        d["trainCode"] = inputs.get("trainCode")
        d["trainConfig"] = inputs.get("trainConfig")
        d["runtime"] = inputs.get("runtime")
    elif node == "evaluate":
        t_key = current_keys.get("train")
        d["trainArtifact"] = cache["train"][t_key]["artifactDigest"] if t_key and t_key in cache.get("train", {}) else None
        d["canonicalData"] = inputs.get("canonicalData")
        d["evaluateCode"] = inputs.get("evaluateCode")
        d["evaluateConfig"] = inputs.get("evaluateConfig")
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

@app.route('/', methods=['GET', 'POST'])
def health_check():
    return "Server is up and running!", 200

# -----------------------------------------
# NEW ENDPOINT: /pipeline
# -----------------------------------------
@app.route('/pipeline', methods=['POST'], strict_slashes=False)
def pipeline():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_REQUEST"}), 409

    if not isinstance(req, dict): return jsonify({"error": "INVALID_REQUEST"}), 409
    
    sess = req.get("session")
    rev = req.get("revision")
    inputs = req.get("inputs")
    events = req.get("events")
    
    if not isinstance(sess, str) or not sess: return jsonify({"error": "INVALID_REQUEST"}), 409
    if not is_safe_pos_int(rev): return jsonify({"error": "INVALID_REQUEST"}), 409
    if not isinstance(inputs, dict): return jsonify({"error": "INVALID_REQUEST"}), 409
    
    req_input_keys = ["generation", "checksum", "canonicalData", "prepareCode", "prepareConfig", "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig", "schemaDigest", "publishConfig"]
    for k in req_input_keys:
        if type(inputs.get(k)) is not str or not inputs[k]:
            return jsonify({"error": "INVALID_REQUEST"}), 409
            
    if not isinstance(events, list): return jsonify({"error": "INVALID_REQUEST"}), 409

    if sess not in pipeline_state:
        pipeline_state[sess] = {
            "session": sess, "revision": rev, "inputs": inputs,
            "cache": {n: {} for n in DAG_NODES},
            "node_states": {}, "event_history": {}
        }
    else:
        state = pipeline_state[sess]
        if rev > state["revision"]:
            state["revision"] = rev
            state["inputs"] = inputs
            state["node_states"] = {}
        elif rev == state["revision"]:
            if state["inputs"] != inputs:
                return jsonify({"error": "REVISION_CONFLICT"}), 409

    state = pipeline_state[sess]
    batch_node_states = {k: dict(v) for k, v in state["node_states"].items()}
    batch_cache = {n: {k: dict(v) for k, v in state["cache"][n].items()} for n in DAG_NODES}
    batch_history = dict(state["event_history"])
    
    accepted = []
    ignored = []
    
    current_keys = compute_keys(state["inputs"], batch_cache)
    
    for e in events:
        if not (isinstance(e, dict) and set(e.keys()) == {"eventId", "revision", "node", "attempt", "status", "key", "artifactDigest", "receiptId"} and isinstance(e["eventId"], str) and is_safe_pos_int(e["revision"]) and isinstance(e["node"], str) and is_safe_pos_int(e["attempt"]) and e["status"] in ["started", "succeeded", "retryable_failed", "terminal_failed"] and isinstance(e["key"], str) and (e["artifactDigest"] is None or isinstance(e["artifactDigest"], str)) and (e["receiptId"] is None or isinstance(e["receiptId"], str))):
            return jsonify({"error": "INVALID_EVENT"}), 409
            
        c_json = canon_json(e)
        eid = e["eventId"]
        
        if eid in batch_history:
            if batch_history[eid] == c_json:
                if eid not in accepted: ignored.append(eid)
                continue
            else:
                return jsonify({"error": "EVENT_ID_CONFLICT"}), 409
                
        if e["revision"] != state["revision"]: ignored.append(eid); continue
        if e["node"] not in DAG_NODES: ignored.append(eid); continue
        if e["key"] != current_keys[e["node"]]: ignored.append(eid); continue
            
        if e["status"] == "succeeded":
            if not e["artifactDigest"]: ignored.append(eid); continue
            if e["node"] in ["register", "publish"]:
                if e["receiptId"] != f"receipt:{e['node']}:{e['key']}": ignored.append(eid); continue
            elif e["receiptId"] is not None: ignored.append(eid); continue
        else:
            if e["artifactDigest"] is not None or e["receiptId"] is not None: ignored.append(eid); continue
                
        node = e["node"]
        key = e["key"]
        cached = batch_cache[node].get(key)
        ns = batch_node_states.get(node)
        
        action = None
        
        if cached:
            if e["status"] == "succeeded":
                if e["artifactDigest"] != cached["artifactDigest"]: return jsonify({"error": "EVIDENCE_CONFLICT"}), 409
                else: action = "ignore"
            else:
                return jsonify({"error": "STATUS_CONFLICT"}), 409
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
            accepted.append(eid)
            batch_history[eid] = c_json
            if e["status"] == "succeeded":
                batch_cache[node][key] = {"artifactDigest": e["artifactDigest"], "eventId": eid, "receiptId": e["receiptId"]}
                if node in batch_node_states: del batch_node_states[node]
                current_keys = compute_keys(state["inputs"], batch_cache)
            else:
                batch_node_states[node] = {"status": e["status"], "attempt": e["attempt"], "eventId": eid, "key": key}
        else:
            ignored.append(eid)
            
    state["node_states"] = batch_node_states
    state["cache"] = batch_cache
    state["event_history"] = batch_history
    
    current_keys = compute_keys(state["inputs"], state["cache"])
    nodes_resp = []
    ancestor_terminal = False
    
    for node in DAG_NODES:
        ckey = current_keys[node]
        cached = state["cache"][node].get(ckey) if ckey else None
        ns = state["node_states"].get(node)
        deps = get_deps(node, state["inputs"], state["cache"], ckey, current_keys)
        action_str, reason, trigger = "", [], []
        
        if cached:
            action_str, reason, trigger = "reuse", ["CACHE_HIT"], [cached["eventId"]]
        else:
            if ckey is not None:
                if ns:
                    if ns["status"] == "started": action_str, reason, trigger = "block", ["RUNNING"], [ns["eventId"]]
                    elif ns["status"] == "terminal_failed":
                        action_str, reason, trigger = "block", ["TERMINAL_FAILURE"], [ns["eventId"]]
                        ancestor_terminal = True
                    elif ns["status"] == "retryable_failed": action_str, reason, trigger = "rerun", ["RETRYABLE_FAILURE"], []
                else:
                    action_str, reason, trigger = "rerun", ["CACHE_MISS"], []
            else:
                action_str = "block"
                reason = ["UPSTREAM_TERMINAL"] if ancestor_terminal else ["UPSTREAM_PENDING"]
                trigger = []
                
        nodes_resp.append({
            "node": node, "action": action_str, "reasonCodes": reason,
            "dependencyDigests": deps, "triggeringEventIds": trigger
        })
        
    return jsonify({
        "revision": state["revision"], "acceptedEventIds": accepted,
        "ignoredEventIds": ignored, "nodes": nodes_resp
    })

# -----------------------------------------
# /quantize (Candidate Admission)
# -----------------------------------------
def is_finite_num(x): return type(x) in (int, float) and not isinstance(x, bool) and math.isfinite(x)
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

# -----------------------------------------
# PREVIOUS ROUTES OMITTED FROM THIS VISUAL BLOCK FOR BREVITY
# (They remain perfectly intact in your actual file, just paste this above them)
# -----------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
