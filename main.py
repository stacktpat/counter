import cv2 as cv
from ultralytics import YOLO , solutions

def main():
    model = YOLO("weights/yolov8n.pt")
    cap = cv.VideoCapture( "vids/bg.mp4" )

    assert(cap.isOpened(), "Error opening video file")
    
    counter = solutions.ObjectCounter(
        view_img = True,
        reg_pts = [(0, 300), (1280, 300)],
        names = model.names, 
        draw_tracks = True,
        line_thickness=1
    )

    while cap.isOpened():
        valid , im0 = cap.read()


        if not valid:
            print("Video end")
            break

        im0 = cv.resize(im0, (1280 , 720) )

        tracks = model.track( im0, persist = True , show = False , classes = [ 0 ])
        im0 = counter.start_counting( im0 , tracks )
        
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    print( counter )
    print("It's take so munch time to start ????")
    pass

if __name__ == '__main__':
    main()
