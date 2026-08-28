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

@app.route('/', methods=['GET', 'POST'])
def health_check():
    return "Server is up and running!", 200

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
    return type(x) is int and not isinstance(x, bool) and 0 < x <= 9007199254740991

def is_finite_num(x):
    return type(x) in (int, float) and not isinstance(x, bool) and math.isfinite(x)

# -----------------------------------------
# NEW ENDPOINT: /adapt (Interventions & PEFT)
# -----------------------------------------
@app.route('/adapt', methods=['POST'], strict_slashes=False)
def adapt():
    try: req = request.get_json(force=True)
    except Exception: return jsonify({"error": "INVALID_INPUT"}), 400

    if type(req) is not dict: return jsonify({"error": "INVALID_INPUT"}), 400
    op = req.get("operation")
    if op not in ["choose", "repair"]: return jsonify({"error": "INVALID_INPUT"}), 400

    if op == "choose":
        policy = req.get("policy")
        cands = req.get("candidates")
        
        if type(policy) is not dict or type(cands) is not list:
            return jsonify({"error": "INVALID_INPUT"}), 400
            
        req_names = ["prompt_only", "retrieval", "lora", "qlora"]
        cand_map = {}
        for c in cands:
            if type(c) is dict and "name" in c and type(c["name"]) is str:
                cand_map[c["name"]] = cand_map.get(c["name"], 0) + 1
        
        for n in req_names:
            if cand_map.get(n, 0) != 1:
                return jsonify({"error": "INVALID_INPUT"}), 400
                
        p_mq = policy.get("minQuality")
        p_fr = policy.get("freshnessRequired")
        p_ml = policy.get("maxLatencyMs")
        p_mm = policy.get("maxMemoryMb")
        p_le = policy.get("maxLabeledExamples")
        p_mt = policy.get("maxTotalCost")
        p_hr = policy.get("horizonRequests")
        
        if not (is_finite_num(p_mq) and 0 <= p_mq <= 1 and type(p_fr) is bool and
                is_finite_num(p_ml) and p_ml >= 0 and is_finite_num(p_mm) and p_mm >= 0 and
                is_safe_int(p_le) and is_finite_num(p_mt) and p_mt >= 0 and is_safe_int(p_hr)):
            return jsonify({"error": "INVALID_INPUT"}), 400

        total_costs = {}
        reason_codes = {n: set() for n in req_names}
        eligible = []

        cand_dicts = {c["name"]: c for c in cands if type(c) is dict and c.get("name") in req_names}
        
        for n in req_names:
            c = cand_dicts[n]
            avail = c.get("available")
            qual = c.get("quality")
            fresh = c.get("freshness")
            lat = c.get("latencyMs")
            mem = c.get("memoryMb")
            lab = c.get("labeledExamples")
            otc = c.get("oneTimeCost")
            rc = c.get("recurringCost")
            
            if not (type(avail) is bool and is_finite_num(qual) and 0 <= qual <= 1 and
                    type(fresh) is bool and is_finite_num(lat) and lat >= 0 and
                    is_finite_num(mem) and mem >= 0 and is_safe_int(lab) and
                    is_finite_num(otc) and otc >= 0 and is_finite_num(rc) and rc >= 0):
                reason_codes[n].add("INVALID_INPUT")
                total_costs[n] = 0.0
                continue
                
            cost = round(otc + p_hr * rc, 12)
            total_costs[n] = cost
            
            if not avail: reason_codes[n].add("UNAVAILABLE")
            if qual < p_mq: reason_codes[n].add("QUALITY_FLOOR")
            if p_fr and not fresh: reason_codes[n].add("FRESHNESS_REQUIRED")
            if lat > p_ml: reason_codes[n].add("LATENCY_LIMIT")
            if mem > p_mm: reason_codes[n].add("MEMORY_LIMIT")
            if lab > p_le: reason_codes[n].add("DATA_LIMIT")
            if cost > p_mt: reason_codes[n].add("COST_LIMIT")
            
            if not reason_codes[n]:
                eligible.append(n)
                
        out_codes = {n: sorted(list(reason_codes[n]), key=lambda x: x.encode('utf-8')) for n in req_names}
        selected = eligible[0] if eligible else None
        
        return jsonify({
            "selected": selected,
            "eligible": eligible,
            "totalCosts": total_costs,
            "reasonCodes": out_codes
        })

    if op == "repair":
        reason_codes = set()
        
        toks = req.get("tokens")
        tok_valid = True
        labels = []
        if type(toks) is not list or not toks:
            tok_valid = False
            reason_codes.add("INVALID_TOKEN")
        else:
            for t in toks:
                if type(t) is not dict: tok_valid = False; break
                tid = t.get("id"); r = t.get("role"); p = t.get("padding"); txt = t.get("text")
                if not is_safe_int(tid) or r not in ["system", "user", "assistant"] or type(p) is not bool or type(txt) is not str:
                    tok_valid = False; break
            
            if not tok_valid:
                reason_codes.add("INVALID_TOKEN")
                labels = [-100] * len(toks) if type(toks) is list else []
            else:
                for t in toks:
                    if t["role"] == "assistant" and t["padding"] is False:
                        labels.append(t["id"])
                    else:
                        labels.append(-100)

        t_apps = req.get("templateApplications")
        t_pass = True
        if t_apps != 1:
            t_pass = False
            reason_codes.add("CHAT_TEMPLATE_COUNT")

        params = req.get("parameters")
        allowed_t = req.get("allowedTargets")
        peft_pass = True
        trainable = []
        trainable_count = 0
        
        if type(params) is not list or type(allowed_t) is not list or not allowed_t:
            peft_pass = False; reason_codes.add("INVALID_PARAMETER")
        else:
            tgt_set = set()
            tgt_valid = True
            for at in allowed_t:
                if type(at) is not str or not at or at in tgt_set:
                    tgt_valid = False; break
                tgt_set.add(at)
                
            p_names = set()
            p_valid = True
            for p in params:
                if type(p) is not dict: p_valid = False; break
                nm = p.get("name"); tgt = p.get("target"); ne = p.get("numel")
                if type(nm) is not str or type(tgt) is not str or not is_safe_pos_int(ne): p_valid = False; break
                if nm in p_names: p_valid = False; break
                p_names.add(nm)
                
            if not tgt_valid or not p_valid:
                peft_pass = False; reason_codes.add("INVALID_PARAMETER")
            else:
                for p in params:
                    nm = p.get("name"); tgt = p.get("target"); ne = p.get("numel")
                    if tgt in tgt_set and (nm.endswith(".lora_A.weight") or nm.endswith(".lora_B.weight")):
                        trainable.append(nm)
                        trainable_count += ne
                
                if not trainable:
                    peft_pass = False; reason_codes.add("INVALID_PARAMETER")
                else:
                    trainable.sort(key=lambda x: x.encode('utf-8'))

        inf = req.get("inferenceMode")
        if inf is not False: reason_codes.add("INFERENCE_MODE")
        
        dr_eval = req.get("dropoutActiveDuringEval")
        if dr_eval is not False: reason_codes.add("EVAL_DROPOUT_ACTIVE")
        
        t_rows = req.get("trainRowIds")
        e_rows = req.get("evalRowIds")
        if type(t_rows) is not list or type(e_rows) is not list or not t_rows or not e_rows:
            reason_codes.add("EVAL_LEAKAGE")
        else:
            t_set = set()
            t_ok = True
            for r in t_rows:
                if type(r) is not str or not r or r in t_set: t_ok = False; break
                t_set.add(r)
            e_set = set()
            e_ok = True
            for r in e_rows:
                if type(r) is not str or not r or r in e_set: e_ok = False; break
                e_set.add(r)
            if not t_ok or not e_ok or not t_set.isdisjoint(e_set):
                reason_codes.add("EVAL_LEAKAGE")

        af = req.get("artifactFiles")
        adapter_files = []
        if type(af) is not list:
            reason_codes.add("ADAPTER_FILE_SET")
        else:
            if "pytorch_model.bin" in af or "model.safetensors" in af:
                reason_codes.add("FULL_MODEL_ARTIFACT")
            elif set(af) != {"adapter_config.json", "adapter_model.safetensors"} or len(af) != 2:
                reason_codes.add("ADAPTER_FILE_SET")
            else:
                adapter_files = sorted(af, key=lambda x: x.encode('utf-8'))

        br = req.get("baseRevision")
        if type(br) is not str or not re.match(r'^[a-f0-9]{40}$', br):
            reason_codes.add("MUTABLE_BASE_REVISION")
            
        dd = req.get("datasetDigest")
        cd = req.get("codeDigest")
        cfd = req.get("configDigest")
        ed = req.get("expectedDigests")
        
        if (type(ed) is not dict or
            type(dd) is not str or not re.match(r'^[a-f0-9]{64}$', dd) or dd != ed.get("dataset") or
            type(cd) is not str or not re.match(r'^[a-f0-9]{64}$', cd) or cd != ed.get("code") or
            type(cfd) is not str or not re.match(r'^[a-f0-9]{64}$', cfd) or cfd != ed.get("config")):
            reason_codes.add("LINEAGE_MISMATCH")

        mb = req.get("microBatch")
        ga = req.get("gradientAccumulation")
        rep = req.get("replicas")
        eeb = req.get("expectedEffectiveBatch")
        if not (is_safe_pos_int(mb) and is_safe_pos_int(ga) and is_safe_pos_int(rep) and is_safe_pos_int(eeb) and mb * ga * rep == eeb):
            reason_codes.add("EFFECTIVE_BATCH_MISMATCH")

        chk = req.get("checkpoint")
        if type(chk) is not dict or set(chk.keys()) != {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}:
            reason_codes.add("INCOMPLETE_CHECKPOINT")

        uw = req.get("uninterruptedWeights")
        rw = req.get("resumedWeights")
        rt = req.get("resumeTolerance")
        
        if (type(uw) is not list or not uw or type(rw) is not list or not rw or
            len(uw) != len(rw) or not is_finite_num(rt) or rt < 0):
            reason_codes.add("RESUME_DIVERGENCE")
        else:
            w_ok = True
            for u, r in zip(uw, rw):
                if not is_finite_num(u) or not is_finite_num(r) or abs(u - r) > rt:
                    w_ok = False; break
            if not w_ok:
                reason_codes.add("RESUME_DIVERGENCE")
                
        eval_iso = "EVAL_LEAKAGE" not in reason_codes
        eval_det = "INFERENCE_MODE" not in reason_codes and "EVAL_DROPOUT_ACTIVE" not in reason_codes
        lin_pass = "MUTABLE_BASE_REVISION" not in reason_codes and "LINEAGE_MISMATCH" not in reason_codes
        chk_comp = "INCOMPLETE_CHECKPOINT" not in reason_codes
        res_pass = "RESUME_DIVERGENCE" not in reason_codes

        return jsonify({
            "labels": labels,
            "templatePass": t_pass,
            "trainableParams": trainable,
            "trainableCount": trainable_count,
            "peftConfigPass": peft_pass,
            "adapterFiles": adapter_files,
            "checkpointComplete": chk_comp,
            "lineagePass": lin_pass,
            "evalIsolated": eval_iso,
            "evaluationDeterministic": eval_det,
            "resumePass": res_pass,
            "reasonCodes": sorted(list(reason_codes), key=lambda x: x.encode('utf-8'))
        })

# -----------------------------------------
# PREVIOUS ROUTES (DO NOT REMOVE)
# -----------------------------------------
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
