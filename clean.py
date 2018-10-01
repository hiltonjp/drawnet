import cv2
import numpy as np
import os

from numpy import newaxis as axis
from screeninfo import get_monitors


class ExitException(BaseException):
    """Allows user to escape from program"""
    pass


class ImageCleaner:
    """ Class for removing background elements from an image dataset

    Static Variables:
        supported_files:  a set of common image file extensions (for sorting out non-images from folder search)

        drawing: whether or not the user is drawing on the canvas
        draw_bg: whether or not the user is drawing foreground or background hints

        image: The crop of the current image
        mask: The mask of the current image
        canvas: The drawing canvas
        preview: A preview of the extraction results
        background: The background to paste extraction results against
        layout: Side-by-side display of canvas and extraction results

        display_scale: the zoom level of the side-by-side display

        brush_size: the size of the brush used in drawing hints

    """

    ####################################################################################################################
    # STATIC VARIABLES                                                                                                 #
    ####################################################################################################################

    supported_files = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')

    # Drawing logic
    drawing = False     # True if mouse down
    draw_bg = True      # True bg drawing; fg otherwise

    # Drawing canvases
    image = None        # Cropped image (used for redrawing canvas/preview)
    mask = None         #
    canvas = None       # Scratch canvas (left side of display)
    preview = None      # GrabCut preview (right side of display)
    background = None   # background color (used to redraw preview)

    layout = None       # display for drawing hints

    # some scalable values
    display_scale = 1
    brush_size = 16

    ####################################################################################################################
    # INITIALIZATION                                                                                                   #
    ####################################################################################################################

    def __init__(self):
        # for grabcut algorithm
        self._gc_rect = None
        self._bgd_model = None
        self._fgd_model = None
        self._gc_iters = 5

        # window names
        self._roi_window_name = "Select Region of Interest"
        self._gc_window_name = "Draw Hints"

    ####################################################################################################################
    # PUBLIC METHODS                                                                                                   #
    ####################################################################################################################

    def clean_folder(self, src_folder, dst_folder=None):
        """Clean a folder of images"""
        print(dst_folder)
        if dst_folder is None:
            dst_folder = src_folder
        elif not os.path.exists(dst_folder):

            os.makedirs(dst_folder)

        try:

            for root, dirs, files in os.walk(src_folder):
                image_paths = [os.path.join(root, file) for file in files if file.lower().endswith(ImageCleaner.supported_files)]

                for image_path in image_paths:

                    image = cv2.imread(image_path)
                    image = self.clean(image)

                    file = os.path.basename(image_path)
                    name, _ = os.path.splitext(file)
                    cv2.imwrite(os.path.join(dst_folder, name + '_clean.png'), image)

        except ExitException:
            return

    def clean(self, raw_image):
        """Clean one image from the dataset by cropping it and iteratively extracting foreground elements"""

        # Scale of display image (display = 1/scale)
        self.display_scale = 1

        # rect = (x, y, width, height)
        rect = self.__get_roi(raw_image)
        if rect is None:
            return raw_image

        # set image of interest
        self.image = raw_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        self.background = np.full_like(ImageCleaner.image, 255, dtype=np.uint8)

        # initialize "potential FG" rectangle, mask, and model arrays for grabcut algoithm
        self.__init_grabcut()
        self.mask, self._bgd_model, self._fgd_model = cv2.grabCut(
            img=self.image,
            mask=self.mask,
            rect=self._gc_rect,
            bgdModel=self._bgd_model,
            fgdModel=self._fgd_model,
            iterCount=self._gc_iters,
            mode=cv2.GC_INIT_WITH_RECT
        )

        ImageCleaner.canvas = ImageCleaner.image.copy()
        self.redraw()

        cv2.namedWindow(self._gc_window_name)
        cv2.setMouseCallback(self._gc_window_name, ImageCleaner.draw)

        # Display loop
        finished = False
        while not finished:
            cv2.imshow(self._gc_window_name, ImageCleaner.layout)
            finished = self.__key_events(cv2.waitKey(20))

        cv2.destroyAllWindows()

        clean_image = self.preview
        return clean_image

    ####################################################################################################################
    # PRIVATE METHODS                                                                                                  #
    ####################################################################################################################

    def __get_roi(self, image):
        """Prompt user to select a rectangular region of interest"""

        scale = self.__rescale_for_image(image)
        rect = cv2.selectROI(
            windowName=self._roi_window_name,
            img=image[::scale, ::scale],
            showCrosshair=False,
            fromCenter=False
        )

        cv2.destroyWindow(self._roi_window_name)

        if rect == (0, 0, 0, 0):
            return None  # No Selection made

        return rect[0]*scale, rect[1]*scale, rect[2]*scale, rect[3]*scale

    def __init_grabcut(self):
        """Initialize grabcut algorithm variables"""

        ImageCleaner.mask = np.zeros(ImageCleaner.image.shape[:2], dtype=np.uint8)

        frame = 1

        # shape[1] is cols (width), shape[0] is rows (height)
        self._gc_rect = (frame, frame, ImageCleaner.image.shape[1] - frame, ImageCleaner.image.shape[0] - frame)
        self._bgd_model = np.zeros((1, 65), dtype=np.float64)
        self._fgd_model = np.zeros_like(self._bgd_model)

    def __key_events(self, key):
        """ Keyboard events for grabcut window"""
        # for readability
        enter_key = 13
        space_key = 32
        num_key_1 = 49
        num_key_9 = 57
        m_key = 109
        z_key = 122
        x_key = 120
        esc_key = 27

        finished = False

        if key == enter_key:
            # Perform another grabcut and reset window

            cv2.destroyWindow(self._gc_window_name)

            ImageCleaner.mask, self._bgd_model, self._fgd_model = cv2.grabCut(
                img=ImageCleaner.image,
                mask=ImageCleaner.mask,
                rect=None,
                bgdModel=self._bgd_model,
                fgdModel=self._fgd_model,
                iterCount=self._gc_iters,
                mode=cv2.GC_INIT_WITH_MASK
            )

            ImageCleaner.canvas = ImageCleaner.image.copy()
            ImageCleaner.redraw()

            cv2.namedWindow(self._gc_window_name)
            cv2.setMouseCallback(self._gc_window_name, ImageCleaner.draw)

        # Save image, move to next image
        elif key == space_key:
            finished = True

        # Rescale display image
        elif num_key_1 <= key <= num_key_9:
            ImageCleaner.display_scale = key - 48

        # Toggle FG/BG painting
        elif key == m_key:
            ImageCleaner.draw_bg = False if ImageCleaner.draw_bg else True

        # brush size down
        elif key == z_key:
            ImageCleaner.brush_size -= 1 if ImageCleaner.brush_size >= 1 else 0

        # brush size up
        elif key == x_key:
            ImageCleaner.brush_size += 1 if ImageCleaner.brush_size <= 200 else 0

        # Exit program
        elif key == esc_key:
            raise ExitException()

        return finished

    ####################################################################################################################
    # STATIC METHODS                                                                                                   #
    ####################################################################################################################

    @staticmethod
    def __rescale_for_image(image):
        # determine appropriate image scaling automatically
        imheight, imwidth = image.shape[:2]

        monitor = get_monitors()[0]

        # in case 1st monitor is tipped longways
        if monitor.height > monitor.width:
            monitor = get_monitors()[1]

        monwidth, monheight = monitor.width, monitor.height

        scale = 1
        while imwidth > monwidth or imheight > monheight:
            scale += 1
            imwidth, imheight = imwidth//scale, imheight//scale

        ImageCleaner.display_scale = scale
        return scale

    @staticmethod
    def draw(event, x, y, flags, param):
        """Image drawing callback"""
        # Scale x and y up to correct size for mask painting
        # (accounts for display image scaling)
        scale = ImageCleaner.display_scale
        x, y = x*scale, y*scale

        if ImageCleaner.drawing and event == cv2.EVENT_MOUSEMOVE:
            filled = -1
            if ImageCleaner.draw_bg:
                # paint display image
                cv2.circle(
                    img=ImageCleaner.canvas,
                    center=(x, y),
                    radius=ImageCleaner.brush_size,
                    color=(0, 0, 0),
                    thickness=filled,
                )

                # paint temp mask
                cv2.circle(
                    img=ImageCleaner.mask,
                    center=(x, y),
                    radius=ImageCleaner.brush_size,
                    color=0,
                    thickness=filled,
                )
            else:
                # paint display image
                cv2.circle(
                    img=ImageCleaner.canvas,
                    center=(x, y),
                    radius=ImageCleaner.brush_size,
                    color=(255, 0, 0),
                    thickness=filled,
                )

                # paint temp mask
                cv2.circle(
                    img=ImageCleaner.mask,
                    center=(x, y),
                    radius=ImageCleaner.brush_size,
                    color=1,
                    thickness=filled,
                )

            # overlay grabcut results onto a clean background
            ImageCleaner.redraw()

        elif event == cv2.EVENT_LBUTTONDOWN:
            ImageCleaner.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            ImageCleaner.drawing = False

    @staticmethod
    def redraw():
        """Redraw the display"""
        alpha_fg = np.where((ImageCleaner.mask == 2) | (ImageCleaner.mask == 0), 0, 1).astype('uint8')
        alpha_bg = 1 - alpha_fg

        # initialize drawing canvas and grabcut preview
        ImageCleaner.preview = alpha_fg[:, :, axis] * ImageCleaner.image + alpha_bg[:, :, axis] * ImageCleaner.background

        scale = ImageCleaner.display_scale
        ImageCleaner.layout = np.concatenate((ImageCleaner.canvas[::scale, ::scale], ImageCleaner.preview[::scale, ::scale]), axis=1)


if __name__ == '__main__':

    cleaner = ImageCleaner()

    cleaner.clean_folder(
        src_folder='/media/hiltonjp/DATA/drawnet/sample',
        dst_folder='/media/hiltonjp/DATA/drawnet/cut'
        # src_folder='/media/jeff/DATA/drawnet/sample',
        # dst_folder='/media/jeff/DATA/drawnet/cut'
    )
