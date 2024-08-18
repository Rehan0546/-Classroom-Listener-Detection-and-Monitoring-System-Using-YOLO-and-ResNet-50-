import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import tensorflow as tf
from tensorflow.keras import layers, models
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def load_classification_model():
    model = models.Sequential()
    model.add(tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
        pooling=None,
        classes=6,))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(6, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('classification.keras')
    return model


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    Path(save_dir/'single').mkdir(parents=True, exist_ok=True)
    Path(save_dir/'grid').mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    classification_model = load_classification_model() 
    classification_names = ['board','focused','raising hand','sleeping','using phone','writing']
    
    listening = ['Passive','Active','Active','Passive','Passive','Active']

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    frame_num = -1
    
    images_gird = []
    
    gird_single_img_shape = None
    
    for path, img, im0s, vid_cap in dataset:
        frame_num = frame_num+1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid_writer.release()
            cv2.destroyAllWindows()
            break
                

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        types = [0,0]
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir /'single'/ p.name)  # img.jpg
            
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                cropped_img = []
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:
                        xyxy = [int(jh) for jh in xyxy]
                        x1,y1,x2,y2 = xyxy
                        cropped_img.append(cv2.resize(im0[y1:y2,x1:x2],(224,224)))
                        
                if len(cropped_img):
                    y_pred = classification_model.predict(np.array(cropped_img),verbose=0)
                    classes_found = np.argmax(y_pred,axis=1)
                    
                    uniques,counts = np.unique(classes_found,return_counts=True)
                    s = ''
                    for un,co in zip(uniques,counts):
                        s += f"{co} {classification_names[int(un)]}{'s' * (int(co) > 1)}, "  # add to string
                        
                    num=-1
                    for *xyxy, conf, cls in reversed(det):
                        if cls != 0:
                            continue
                        num=num+1
    
                        if save_img or view_img:  # Add bbox to image
                            cls = classes_found[num]
                            conf = y_pred[num][cls]
                            if listening[cls] == 'Active':
                                types[1] = types[1]+ 1
                            else:
                                types[0] = types[0]+ 1
                            # print(label,conf)
                            label = f'{classification_names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            
            
            
            label = f'Passive Listners: {types[0]}'
            
            
            (w, h), _ = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            height, width, channels = im0.shape
            x1 = width-w-10
            
            y1 = height-2*h-10-10
            # Prints the text.    
            cv2.rectangle(im0, (x1, y1-2), (x1 + w, y1+h+2 ), (0,0,0), -1)
            cv2.putText(im0, label, (x1, y1+h),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            
            
            label = f'Active Listners: {types[1]}'
            
            # (w, h), _ = cv2.getTextSize( label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            height, width, channels = im0.shape
            # x1 = width-w-10
            # y1 = height-h-10
            y1 = y1+h+10-2
            # Prints the text.    
            cv2.rectangle(im0, (x1, y1), (x1 + w, y1+ h+2), (0,0,0), -1)
            cv2.putText(im0, label, (x1, y1+h  ),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            
            
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('show', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            
            
            if frame_num%opt.gird_frame_skipping==0:
                
                if not gird_single_img_shape:
                    gird_single_img_shape = im0.shape[:-1]
                    
                if not im0.shape[:-1] == gird_single_img_shape:
                    im0 = cv2.resize(im0,(gird_single_img_shape[1],gird_single_img_shape[0]))
                    
                images_gird.append(im0)
                
                if len(images_gird) == 4:
                    
                    image_row_1 = cv2.hconcat([images_gird[0], images_gird[1]])
                    image_row_2 = cv2.hconcat([images_gird[2], images_gird[3]])
                    grid = cv2.vconcat([image_row_1, image_row_2])
                    
                    images_gird = []
                    # print(p.name)
                    gird_path = str(save_dir /'grid'/ str(p.name+str(frame_num)+".jpg"))  # img.jpg
                    
                    cv2.imwrite(gird_path,grid)
                    
                    cv2.imshow('grid', cv2.resize(grid,None,fx=0.5,fy=0.5))
                    cv2.waitKey(1)  # 1 millisecond
                
                
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='v5lite-s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--gird_frame_skipping', type=int, default=1, help='gird_frame_skipping')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img',default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
