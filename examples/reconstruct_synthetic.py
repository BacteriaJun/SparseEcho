from sparseecho import SparseEchoEngine, compile_query_plan
from sparseecho.config import SceneConfig
from sparseecho.metrics import detection_metrics
from sparseecho.simulator import simulate_capture

plan = compile_query_plan()
capture = simulate_capture(plan, SceneConfig(), seed=7)
result = SparseEchoEngine(plan=plan).process_capture(capture.slots, capture.valid)
print(result.identities)
print(detection_metrics(capture.truth.identities, result.identities))
