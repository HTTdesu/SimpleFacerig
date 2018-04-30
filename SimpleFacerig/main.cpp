
#include <stdlib.h>
#include <math.h>
#include <string>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>

#include <libpng/png.h>
#include <live2d/Live2D.h>
#include <live2d/util/UtSystem.h>
#include <live2d/Live2DModelOpenGL.h>
#include <live2d/util/Json.h>

#include <opencv2/opencv.hpp> 

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>

using namespace std;

live2d::Live2DModelOpenGL* live2DModel;
cv::VideoCapture* camera;
dlib::frontal_face_detector detector;
dlib::shape_predictor pose_model;

// live2d parameters
float x_rotate = 0.0f;
float y_rotate = 0.0f;
float z_rotate = 0.0f;

float left_eye = 1.0f;
float right_eye = 1.0f;

float eyebrow_left = 0.0f;
float eyebrow_right = 0.0f;

float mouth_open = 0.0f;

double meter(double A, double B, double C, double x, double y)
{
	double diversion = A * x + B * y + C;
	return diversion / sqrt(pow(A, 2) + pow(B, 2));
}

double eyebrow_move(dlib::point &p1, dlib::point &p2, double slope, double last, double rate)
{
	double bias = p1.y() - slope * p1.x();
	double distance = sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2));
	double diversion = meter(slope, -1, bias, p2.x(), p2.y());
	double result = (diversion / distance - 0.45) * 6;
	result = last * (1 - rate) + result * rate;
	return result;
}

double eye_open(dlib::point &p1, dlib::point &p2, dlib::point &v11, dlib::point &v12, dlib::point &v21, dlib::point &v22, double last, double rate)
{
	double distance = sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2));
	double slope = (double)(p2.y() - p1.y()) / (double)(p2.x() - p1.x());
	double bias = p2.y() - slope * p2.x();
	double d1 = abs(meter(slope, -1, bias, v11.x(), v11.y()));
	double d2 = abs(meter(slope, -1, bias, v12.x(), v12.y()));
	double diversion = d1 > d2 ? d1 : d2;
	d1 = abs(meter(slope, -1, bias, v21.x(), v21.y()));
	d2 = abs(meter(slope, -1, bias, v22.x(), v22.y()));
	diversion += d1 > d2 ? d1 : d2;
	double ratio = (diversion / distance - 0.18) * 8;
	ratio = ((int)(ratio * 10.0)) / 10.0;
	ratio = last * (1 - rate) + ratio * rate;
	return ratio;
}

void display(void)
{
	cv::Mat buff, temp;
	*camera >> temp;
	cv::GaussianBlur(temp, buff, cv::Size(5, 5), 5, 5);
	cv::flip(buff, temp, 1);

	dlib::cv_image<dlib::bgr_pixel> cimg(temp);
	vector<dlib::rectangle> faces = detector(cimg);
	vector<dlib::full_object_detection> shapes;

	for (unsigned long i = 0; i < faces.size(); ++i)
		shapes.push_back(pose_model(cimg, faces[i]));

	if (!shapes.empty()) {
		for (int i = 0; i < 68; i++) {
			cv::circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(0, 0, 255), -1);
		}

		int div_x = shapes[0].part(16).x() - shapes[0].part(0).x();
		int div_y = shapes[0].part(16).y() - shapes[0].part(0).y();
		double center_x = shapes[0].part(0).x() + div_x / 2.0;
		double center_y = shapes[0].part(0).y() + div_y / 2.0;
		double slope = (double)(div_y) / (double)(div_x);
		double bias = center_y - slope * center_x;
		double x_proj = (slope * (shapes[0].part(30).y() - bias) + shapes[0].part(30).x()) / (1 + pow(slope, 2));
		double y_proj = slope * x_proj + bias;
		double diversion = sqrt(pow(x_proj - shapes[0].part(0).x(), 2) + pow(y_proj - shapes[0].part(0).y(), 2));
		double distance = sqrt(pow(shapes[0].part(16).x() - shapes[0].part(0).x(), 2) + pow(shapes[0].part(16).y() - shapes[0].part(0).y(), 2));

		double rate = 0.5;
		// Ax+By+C/sqrt(A^2+B^2)
		x_rotate = x_rotate * (1 - rate) + asin(diversion / distance - 0.5) * 3.14 * 40.0 * rate;

		// nose to eye around 1/6 head
		diversion = meter(slope, -1, bias, shapes[0].part(30).x(), shapes[0].part(30).y());
		diversion = diversion + 1.0 / 6 * distance;
		y_rotate = y_rotate * (1 - rate) + asin(diversion / distance) * 3.14 * 40.0 * rate;

		z_rotate = z_rotate * (1 - rate) + atan(slope) * 3.14 * 40 * rate;

		// eye
		left_eye = eye_open(shapes[0].part(36), shapes[0].part(39), shapes[0].part(37), shapes[0].part(38), shapes[0].part(40), shapes[0].part(41), left_eye, rate);
		right_eye = eye_open(shapes[0].part(42), shapes[0].part(45), shapes[0].part(43), shapes[0].part(44), shapes[0].part(46), shapes[0].part(47), right_eye, rate);

		// eyebrow
		eyebrow_left = eyebrow_move(shapes[0].part(17), shapes[0].part(19), slope, eyebrow_left, rate);
		eyebrow_right = eyebrow_move(shapes[0].part(26), shapes[0].part(24), slope, eyebrow_right, rate);

		// mouth
		diversion = sqrt(pow(shapes[0].part(62).x() - shapes[0].part(66).x(), 2) + pow(shapes[0].part(62).y() - shapes[0].part(66).y(), 2));
		distance = sqrt(pow(shapes[0].part(60).x() - shapes[0].part(64).x(), 2) + pow(shapes[0].part(60).y() - shapes[0].part(64).y(), 2));
		mouth_open = (diversion / distance - 0.15) * 2;
	}
	else
	{
		x_rotate = 0.0f;
		y_rotate = 0.0f;
		z_rotate = 0.0f;
		left_eye = 1.0f;
		right_eye = 1.0f;
		eyebrow_left = 0.0f;
		eyebrow_right = 0.0f;
		mouth_open = 0.0f;
	}

	glClear(GL_COLOR_BUFFER_BIT);
	//double t = (live2d::UtSystem::getUserTimeMSec() / 1000.0) * 2 * 3.14;
	//live2DModel->setParamFloat("PARAM_ANGLE_Z", (float)(30 * sin(t / 3.0)));
	live2DModel->setParamFloat("PARAM_ANGLE_X", x_rotate);
	live2DModel->setParamFloat("PARAM_ANGLE_Y", y_rotate);
	live2DModel->setParamFloat("PARAM_ANGLE_Z", z_rotate);

	live2DModel->setParamFloat("PARAM_EYE_L_OPEN", left_eye);
	live2DModel->setParamFloat("PARAM_EYE_R_OPEN", right_eye);

	live2DModel->setParamFloat("PARAM_BROW_L_Y", eyebrow_left);
	live2DModel->setParamFloat("PARAM_BROW_R_Y", eyebrow_right);

	live2DModel->setParamFloat("PARAM_MOUTH_OPEN_Y", mouth_open);

	live2DModel->update();
	live2DModel->draw();

	//Display it all on the screen  
	cv::imshow("Feature points", temp);
	glFlush();
}

void timer(int value) {


	glutPostRedisplay();
	glutTimerFunc(30, timer, 0);
}

int loadGLTexture(const char* path)
{
	unsigned int id;

	FILE            *fp;
	png_structp     png_ptr;
	png_infop       info_ptr;
	unsigned int   width, height;
	int             bit_depth, color_type, interlace_type;
	unsigned char   *image;

	errno_t error;

	if ((error = fopen_s(&fp, path, "rb")) != 0) {
		printf("file not exists!");
		return -1;
	}

	png_ptr = png_create_read_struct(
		PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info_ptr = png_create_info_struct(png_ptr);

	png_init_io(png_ptr, fp);
	png_read_info(png_ptr, info_ptr);
	png_get_IHDR(png_ptr, info_ptr, &width, &height,
		&bit_depth, &color_type, &interlace_type,
		NULL, NULL);
	int rb = png_get_rowbytes(png_ptr, info_ptr);
	image = new unsigned char[height * rb];
	unsigned char **recv = new unsigned char*[height];
	for (int i = 0; i < height; i++)
		recv[i] = &image[i * rb];
	png_read_image(png_ptr, recv);
	png_read_end(png_ptr, info_ptr);

	//premultiplied alpha


	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);
	delete[] recv;


	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &id);

	glBindTexture(GL_TEXTURE_2D, id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	/*glTexImage2D(
	GL_TEXTURE_2D , 0 , GL_RGBA , width , height ,
	0 , GL_RGBA , GL_UNSIGNED_BYTE , image
	);*/
	gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, width, height, GL_RGBA, GL_UNSIGNED_BYTE, image);
	return id;
}

void Live2DInit(void)
{
	glClearColor(0.0, 0.0, 1.0, 1.0);

	std::string path = "res/Epsilon/";
	live2DModel = live2d::Live2DModelOpenGL::loadModel((char*)((path + "Epsilon.moc").c_str()));

	const char* TEXTURES[] = {
		(char*)((path + "Epsilon.2048/texture_00.png").c_str()),
		NULL ,
	};

	for (int i = 0; i < 1000; i++) {
		if (!TEXTURES[i]) break;

		int tex = loadGLTexture(TEXTURES[i]);

		if (tex < 0) printf("failed");
		live2DModel->setTexture(i, tex);
	}
	live2DModel->setPremultipliedAlpha(false);
}

void OpenCVInit(void)
{
	camera = new cv::VideoCapture(0);
	if (!camera->isOpened())
	{
		cerr << "Unable to connect to camera" << endl;
	}

	detector = dlib::get_frontal_face_detector();
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
}

void resize(int w, int h)
{
	glViewport(0, 0, w, h);

	glLoadIdentity();

	float aspect = (float)w / h;
	float sx = 2.0 / live2DModel->getCanvasWidth();
	float sy = -2.0 / live2DModel->getCanvasWidth() * aspect;
	float x = -1;
	float y = 1;
	float matrix[] = {
		sx , 0 , 0 , 0 ,
		0 , sy ,0 , 0 ,
		0 , 0 , 1 , 0 ,
		x , y , 0 , 1
	};


	live2DModel->setMatrix(matrix);
}


int main(int argc, char *argv[])
{
	live2d::Live2D::init();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(600, 600);
	glutCreateWindow("Model");
	Live2DInit();
	OpenCVInit();
	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	glutTimerFunc(100, timer, 0);
	glutMainLoop();
	return 0;
}

