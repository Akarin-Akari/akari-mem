from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

print('=== Embedding models (BGE-M3 check) ===')
models = TextEmbedding.list_supported_models()
bge = [m for m in models if 'bge-m3' in m.get('model', '').lower()]
for m in bge:
    print(f"  {m['model']} | dim={m.get('dim')} | size={m.get('size_in_GB')}GB")

print()
print('=== Reranker models (bge-reranker check) ===')
rmodels = TextCrossEncoder.list_supported_models()
for m in rmodels:
    if 'bge' in m.get('model', '').lower():
        print(f"  {m['model']} | size={m.get('size_in_GB')}GB")
