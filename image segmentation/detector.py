from mmdet.apis import DetInferencer

# Initialize the model
def initialize(cfg_path, ckpt_path, device) :
    return DetInferencer(
        model = cfg_path,
        weights = ckpt_path,
        device = device,
        show_progress = False
    )
 
def inference(input_path, model) :
    result = model(
        inputs = input_path,
        out_dir = None,
        no_save_vis = True,
        pred_score_thr = 0.75
    )

    predictions = result['predictions'][0]
    valid_idx = [i for i, score in enumerate(predictions['scores']) if score >= 0.75]
    valid_box = [predictions['bboxes'][i] for i in valid_idx]

    # Return the bbox result
    return valid_box
