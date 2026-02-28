#!/bin/bash
# Speculative decoding benchmark for MXFP4 flash attention
# Tests different n-gram configurations and measures acceptance rates
set -euo pipefail

IMAGE="local/llama.cpp:full-cuda"
MODEL="/models/spectralyst/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE-GGUF/Qwen3-Coder-30B-A3B-Instruct-MXFP4_MOE.gguf"
PORT=8085
SERVER_URL="http://localhost:${PORT}"

# Common server args (everything except spec params)
COMMON_ARGS=(
    "--model" "${MODEL}"
    "--host" "0.0.0.0" "--port" "8085"
    "--threads" "8" "--threads-batch" "8"
    "--cpu-range" "0-7" "--cpu-range-batch" "0-7" "--cpu-strict" "1"
    "--flash-attn" "on"
    "--cache-type-k" "mxfp4" "--cache-type-v" "mxfp4"
    "--ctx-size" "32768"
    "--parallel" "1"
    "--jinja"
)

# Test payloads
PAYLOAD_CODE='{
  "model": "test",
  "messages": [
    {"role": "system", "content": "You are a coding assistant. When asked to modify code, output the complete file contents."},
    {"role": "user", "content": "Create a simple Flask REST API with CRUD operations for a User model. The User has id, name, and email fields. Use a simple in-memory dictionary as storage."},
    {"role": "assistant", "content": "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\nusers = {}\nnext_id = 1\n\n@app.route(\"/users\", methods=[\"GET\"])\ndef get_users():\n    return jsonify(list(users.values())), 200\n\n@app.route(\"/users/<int:user_id>\", methods=[\"GET\"])\ndef get_user(user_id):\n    user = users.get(user_id)\n    if not user:\n        return jsonify({\"error\": \"User not found\"}), 404\n    return jsonify(user), 200\n\n@app.route(\"/users\", methods=[\"POST\"])\ndef create_user():\n    global next_id\n    data = request.get_json()\n    if not data or \"name\" not in data or \"email\" not in data:\n        return jsonify({\"error\": \"Missing required fields\"}), 400\n    user = {\"id\": next_id, \"name\": data[\"name\"], \"email\": data[\"email\"]}\n    users[next_id] = user\n    next_id += 1\n    return jsonify(user), 201\n\n@app.route(\"/users/<int:user_id>\", methods=[\"PUT\"])\ndef update_user(user_id):\n    user = users.get(user_id)\n    if not user:\n        return jsonify({\"error\": \"User not found\"}), 404\n    data = request.get_json()\n    if \"name\" in data:\n        user[\"name\"] = data[\"name\"]\n    if \"email\" in data:\n        user[\"email\"] = data[\"email\"]\n    return jsonify(user), 200\n\n@app.route(\"/users/<int:user_id>\", methods=[\"DELETE\"])\ndef delete_user(user_id):\n    if user_id not in users:\n        return jsonify({\"error\": \"User not found\"}), 404\n    del users[user_id]\n    return jsonify({\"message\": \"User deleted\"}), 200\n\nif __name__ == \"__main__\":\n    app.run(debug=True)\n```"},
    {"role": "user", "content": "Now add a Product model with id, name, price, and description fields. Add the same CRUD operations for products. Keep the existing user code."}
  ],
  "max_tokens": 1500,
  "temperature": 0
}'

PAYLOAD_SIMPLE='{
  "model": "test",
  "messages": [{"role": "user", "content": "Write a Python function that sorts a list using merge sort. Include type hints."}],
  "max_tokens": 500,
  "temperature": 0
}'

extract_stats() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
dn = t.get('draft_n', 0)
da = t.get('draft_n_accepted', 0)
tokens = d['usage']['completion_tokens']
tps = t.get('predicted_per_second', 0)
accept = da/max(dn,1)*100 if dn else 0
print(f'{tokens}\t{dn}\t{da}\t{accept:.1f}\t{tps:.1f}')
"
}

wait_for_server() {
    for i in $(seq 1 120); do
        if curl --silent --fail "${SERVER_URL}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "  TIMEOUT"
    return 1
}

cleanup() {
    docker stop spec-bench > /dev/null 2>&1 || true
    docker rm spec-bench > /dev/null 2>&1 || true
}

run_config() {
    local name="$1"
    shift
    local spec_args=("$@")

    echo ""
    echo "=== ${name} ==="
    echo "  Starting server..."

    cleanup

    # Start server in background with proper entrypoint
    docker run --rm --detach \
        --name spec-bench \
        --gpus all \
        --publish "${PORT}:8085" \
        --volume /home/tim/.lmstudio/models:/models:ro \
        --entrypoint /app/llama-server \
        "${IMAGE}" \
        "${COMMON_ARGS[@]}" "${spec_args[@]}" \
        > /dev/null

    if ! wait_for_server; then
        echo "  Server failed to start, checking logs..."
        docker logs spec-bench --tail 20
        cleanup
        return 1
    fi

    echo -e "  test\t\t\ttokens\tdraft_n\tdraft_a\tacc%\tt/s"

    echo -n "  code-crud-warm\t"
    echo "${PAYLOAD_CODE}" | curl --silent --max-time 120 "${SERVER_URL}/v1/chat/completions" \
        --header "Content-Type: application/json" --data @- | extract_stats

    echo -n "  code-crud-replay\t"
    echo "${PAYLOAD_CODE}" | curl --silent --max-time 120 "${SERVER_URL}/v1/chat/completions" \
        --header "Content-Type: application/json" --data @- | extract_stats

    echo -n "  merge-sort-warm\t"
    echo "${PAYLOAD_SIMPLE}" | curl --silent --max-time 60 "${SERVER_URL}/v1/chat/completions" \
        --header "Content-Type: application/json" --data @- | extract_stats

    echo -n "  merge-sort-replay\t"
    echo "${PAYLOAD_SIMPLE}" | curl --silent --max-time 60 "${SERVER_URL}/v1/chat/completions" \
        --header "Content-Type: application/json" --data @- | extract_stats

    cleanup
    sleep 3
}

echo "============================================"
echo "  MXFP4 N-gram Speculative Decoding Bench"
echo "============================================"
echo ""

# Ensure clean state
cleanup

# Baseline — no spec decoding at all (just omit spec flags)
run_config "BASELINE (no spec)"

# Config A: Current default
run_config "A: ngram-simple n=8 m=32 draft=32" \
    "--spec-type" "ngram-simple" \
    "--spec-ngram-size-n" "8" "--spec-ngram-size-m" "32" "--draft" "32"

# Config B: Shorter drafts
run_config "B: ngram-simple n=8 m=8 draft=8" \
    "--spec-type" "ngram-simple" \
    "--spec-ngram-size-n" "8" "--spec-ngram-size-m" "8" "--draft" "8"

# Config C: map-k4v with short drafts
run_config "C: map-k4v n=8 m=8 draft=8" \
    "--spec-type" "ngram-map-k4v" \
    "--spec-ngram-size-n" "8" "--spec-ngram-size-m" "8" "--draft" "8"

# Config D: Shorter n-gram
run_config "D: map-k4v n=4 m=16 draft=16" \
    "--spec-type" "ngram-map-k4v" \
    "--spec-ngram-size-n" "4" "--spec-ngram-size-m" "16" "--draft" "16"

# Config E: Very short everything
run_config "E: ngram-simple n=4 m=8 draft=8" \
    "--spec-type" "ngram-simple" \
    "--spec-ngram-size-n" "4" "--spec-ngram-size-m" "8" "--draft" "8"

echo ""
echo "=== Benchmark complete ==="
