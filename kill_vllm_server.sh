ps xu | grep -i "vllm" | grep -v "grep" | awk '{print $2}' | xargs kill -9
ps xu | grep -i "kimi" | grep -v "grep" | awk '{print $2}' | xargs kill -9
ps xu | grep -i "import main" | grep "python" | awk '{print $2}' | xargs kill -9
