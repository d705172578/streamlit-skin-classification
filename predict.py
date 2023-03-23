import torch
from torchvision import transforms
from model.model_v8 import ResIUNet
from utils import *
from torch.utils import model_zoo

ienet_url = 'https://huggingface.co/Inubashiri/IENet/resolve/main/v8_isic.pth'
classification_url = 'https://huggingface.co/Inubashiri/IENet/resolve/main/my_model(199).pkl'

classification_model = model_zoo.load_url(classification_url)
classification_model.cuda()

ienet = ResIUNet()
ienet.load_state_dict(model_zoo.load_url(ienet_url))
ienet = ienet.cuda()


def seg_pre_processing(img):
    norm_3D = transforms.Normalize(torch.from_numpy(np.array([0., 0., 0.])).float(), torch.from_numpy(np.array([1., 1., 1.])).float())
    transformed_img = transforms.ToTensor()(img)
    normed_img = norm_3D(transformed_img)
    return torch.unsqueeze(transforms.Resize((384, 384))(normed_img).to(torch.device('cuda:0')), 0)


def seg_post_processing(img, prediction):
    to_pil = transforms.ToPILImage()

    t_pred = torch.repeat_interleave(prediction, 3, dim=0)
    np_pred = prediction[0].cpu().data.numpy()

    mulmask = t_pred * img
    reverse_pred = 1 - t_pred
    np_img = img.cpu().data.numpy().swapaxes(0, 2)

    img_blur = img_filter(cv2.cvtColor((np_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB), f_size=41)
    img_blur = cv2.cvtColor(img_blur.astype(np.float32) / 255, cv2.COLOR_RGB2BGR).swapaxes(0, 2)
    tensor_blur = torch.tensor(img_blur).cuda()

    hard_pred = hard(np_pred)

    locs = np.where(hard_pred == 1.0)
    x0 = np.min(locs[1])
    x1 = np.max(locs[1])
    y0 = np.min(locs[0])
    y1 = np.max(locs[0])

    new_img = to_pil((tensor_blur * reverse_pred + mulmask)[:, y0:y1, x0:x1])
    disease_mask = to_pil(mulmask)
    disease_crop = to_pil(img[:, y0:y1, x0:x1])
    blur_bg = to_pil(tensor_blur * reverse_pred)
    return new_img, disease_mask, disease_crop, blur_bg


def seg_pred(img):
    processed_img = seg_pre_processing(img)
    prediction = ienet(processed_img)
    return seg_post_processing(processed_img[0], prediction[0])


def cls_pre_processing(img):
    seg_img, disease_mask, disease_crop, blur_bg = seg_pred(img)

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transformed_image = transform(seg_img)

    return torch.unsqueeze(transformed_image.to(torch.device('cuda:0')), 0), seg_img, disease_mask, disease_crop, blur_bg

def post_processing(output):        # 后处理
    name = ['光化性角化病', '基底细胞癌', '良性角化病', '皮肤纤维瘤', '黑色素瘤', '黑色素细胞性痣', '血管性皮肤病变']
    values, inds = torch.topk(torch.softmax(output, 1), 3)
    res = [[round(value.item(), 3), name[ind.item()]] for value, ind in zip(values[0], inds[0])]
    return res


def get_pred(img):
    processed_img, seg_img, disease_mask, disease_crop, blur_bg = cls_pre_processing(img)
    output = classification_model(processed_img)
    return post_processing(output), [seg_img, disease_mask, disease_crop, blur_bg]


if __name__ == '__main__':
    src = cv2.imread(r'E:\streamlit_skin\test\akiec1.jpg')
    import time
    start = time.time()
    print(get_pred(src))
    print('use', time.time() - start)
