from .chamfer import compute_chamfer_distance
from .clip import compute_clip_similarity
from .fid_kid import compute_fid_kid
from .iou import compute_iou
from .lpips_ import compute_lpips
from .psnr import compute_psnr
from .ssim import compute_ssim
from enum import Enum
class Metric(Enum):
    CD = 1, #chamder distance
    CLIP = 2,
    FID_KID = 3,
    IOU = 4,
    LPIPS = 5,
    PSNR = 6,
    SSIM = 7
evaluate = {
        Metric.CD: compute_chamfer_distance,
        Metric.CLIP: compute_clip_similarity,
        Metric.FID_KID: compute_fid_kid,
        Metric.IOU: compute_iou,
        Metric.LPIPS: compute_lpips,
        Metric.PSNR: compute_psnr,
        Metric.SSIM: compute_ssim
    }
def reconstruction_evaluate_2_shapes(path1: str,
                     path2: str,
                     metric: Metric,
                     **kwargs):
    
    assert metric in [Metric.CD, Metric.IOU, Metric.PSNR, Metric.SSIM], "Invalid metric"
    if metric == Metric.IOU:
        if "resolution" in kwargs.keys():
            reso = kwargs["resolution"]
            return evaluate[metric](path1, path2, reso)
    return evaluate[metric](path1, path2)

def generation_evaluate_shape(path: str,
                     caption: str,
                     metric: Metric):
    
    assert metric in [Metric.CLIP, Metric.FID_KID], "Invalid metric"
    if metric == Metric.CLIP:
        return evaluate[metric](path, caption)
    elif metric == Metric.FID_KID:
        return compute_fid_kid([])

    
    
def reconstruction_evaluate_set(list_path1: list, list_path2: list, metric: Metric, **kwargs):
    assert len(list_path1) == len(list_path2), "Two set must be the same size"
    sum_value = 0
    for i in range(len(list_path1)):
        if metric == Metric.IOU:
            if "resolution" in kwargs.keys():
                reso = kwargs["resolution"]
                metric_value =  reconstruction_evaluate_2_shapes(list_path1[i], list_path2[i], metric, resolution=reso)
        else:
            metric_value = reconstruction_evaluate_2_shapes(list_path1[i], list_path2[i], metric)
        sum_value += metric_value
    return sum_value / len(list_path1)

def generation_evaluate_set(path_list: list,
                     captions: list,
                     metric: Metric):
    assert len(path_list) == len(captions), "Path list and caption list must be the same size"
    sum_value = 0
    for i in range(len(path_list)):
        metric_value = generation_evaluate_shape(path_list[i], captions[i], metric)

        sum_value += metric_value
    return sum_value / len(path_list)    

    
    