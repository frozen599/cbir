#include "stdafx.h"
#include "Project_(1).h"

//calculate histogram for each image
vector<Mat> calc_Img(Mat srcImg, bool gbin) {
	vector<Mat> channels;
	split(srcImg, channels);
	
	vector<Mat>hists;
	float sranges[] = { 0,256 };
	const float* range = { sranges };
	Mat R_hist;
	Mat G_hist;
	Mat B_hist;
	int histSize;
	if (gbin)
		histSize = 32;
	else
		histSize = 256;

	calcHist(&channels[0], 1, 0, Mat(), R_hist, 1, &histSize, &range, true, false);
	calcHist(&channels[1], 1, 0, Mat(), G_hist, 1, &histSize, &range, true, false);
	calcHist(&channels[2], 1, 0, Mat(), B_hist, 1, &histSize, &range, true, false);

	hists.push_back(R_hist);
	hists.push_back(G_hist);
	hists.push_back(B_hist);

	return hists;
}

//calculate 2 histograms
int calc2Hists(vector<int>h1, vector<int>h2) {
	int sum = 0;
	for (int i = 0; i < h1.size(); i++) {
		if (h1[i] < h2[i]) {
			sum += h1[i];
		}
		else {
			sum += h2[i];
		}
	}
	return sum;
}



Mat kMeanClusterMat(Mat image)
{
	// convert to float & reshape to a [3 x W*H] Mat 
	//  (so every pixel is on a row of it's own)
	Mat data;
	image.convertTo(data, CV_32F);
	data = data.reshape(1, data.total());

	// do kmeans
	Mat labels, centers;
	kmeans(data, 8, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
		KMEANS_PP_CENTERS, centers);

	// reshape both to a single row of Vec3f pixels:
	centers = centers.reshape(3, centers.rows);
	data = data.reshape(3, data.rows);

	// replace pixel values with their center value:
	Vec3f *p = data.ptr<Vec3f>();
	for (size_t i = 0; i < data.rows; i++) {
		int center_id = labels.at<int>(i);
		p[i] = centers.at<Vec3f>(center_id);
	}

	Mat res;
	// back to 2d, and uchar:
	image = data.reshape(3, image.rows);
	image.convertTo(res, CV_8U);

	return res;
}


void colorMatching(vector<Database>& db, Database src)
{

	// rearrange database
	for (int i = 0; i < db.size() - 1; i++)
		for (int j = i + 1; j < db.size(); j++)
			if (db[i].urlImg.compare(db[j].urlImg) > 0)
				swap(db[i], db[j]);

	// calculate query image color matrix
	Mat srcColorMat = kMeanClusterMat(src.srcImg);
	// calculate color image color histogram
	vector<Mat> srcColorMatHist = calc_Img(srcColorMat, false);


	vector<vector<Mat>> dbColorHist;
	// calculate database's images color matrix
	for (int i = 0; i < db.size(); i++)
	{
		Mat temp = imread(db[i].urlImg, IMREAD_COLOR);
		Mat dbColorMat = kMeanClusterMat(temp);
		vector<Mat> dbColorMatHist = calc_Img(dbColorMat, false);
		dbColorHist.push_back(dbColorMatHist);
	}


	double accumulator_1 = 0, accumulator_2 = 0;
	// comapre color histogram and rearrange in descending order
	for (int i = 0; i < srcColorMatHist.size() - 1; i++)
	{
		for (int j = i + 1; j < srcColorMatHist.size(); j++)
		{
			double accumulator_1 = 0, accumulator_2 = 0;

			for (int k = 0; k < 3; k++)
			{
				accumulator_1 += compareHist(srcColorMatHist[k], dbColorHist[i][k], CV_COMP_BHATTACHARYYA);
				accumulator_2 += compareHist(srcColorMatHist[k], dbColorHist[j][k], CV_COMP_BHATTACHARYYA);
			}
			if (accumulator_1 < accumulator_2)
				swap(db[i], db[j]);

		}
	}

}


// calculate edge feature matrix of an image
Mat calcEdgeMat(Mat image)
{
	Mat grayScale;
	// convert to grayscale image
	cvtColor(image, grayScale, COLOR_BGR2GRAY);
	// using Canny to detect edge
	Mat detectEdge, dest;
	Canny(grayScale, detectEdge, 50, 150, 3);
	// convert back to 8UC1
	detectEdge.convertTo(dest, CV_8UC1);
	return dest;
}



// calculate databse edge feature matrices and query image edge matrix
// and then compare histogram of each pair, rearrange in desceding order of 
// matching histogram
void edgeMatching(vector<Database> &objs, Database src)
{

	for (int i = 0; i < objs.size() - 1; i++)
		for (int j = i + 1; j < objs.size(); j++)
			if (objs[i].urlImg.compare(objs[j].urlImg) > 0)
				swap(objs[i], objs[j]);

	// calculate query image 's edge matrix
	Mat srcEdgeMat = calcEdgeMat(src.srcImg);

	vector<Mat> dbEdgeMats;

	// calculate database 's images edge matrix
	for (int i = 0; i < objs.size(); i++)
	{
		Mat objsImage = imread(objs[i].urlImg, IMREAD_COLOR);
		Mat objsEdgeMat = calcEdgeMat(objsImage);
		dbEdgeMats.push_back(objsEdgeMat);
	}

	int histSize = 256;
	Mat tempHist, srcEdgeMatHist;
	vector<Mat> objEdgeHists;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	// histogram of source image's edge matrix
	calcHist(&srcEdgeMat, 1, 0, Mat(), srcEdgeMatHist, 1, &histSize, &histRange, true, false);
	// calculate hists of all edge matrices
	for (int i = 0; i < dbEdgeMats.size(); i++)
	{
		calcHist(&dbEdgeMats[i], 1, 0, Mat(), tempHist, 1, &histSize, &histRange, true, false);
		objEdgeHists.push_back(tempHist);
		tempHist.release();
	}

	// compare each database edge matrix histogram with query image edge's matrix
	// and reorder in descending order according to matching histogram
	for (int i = 0; i < objEdgeHists.size() - 1; i++)
		for (int j = i + 1; j < objEdgeHists.size(); j++)
		{
			double temp1 = compareHist(srcEdgeMatHist, objEdgeHists[i], CV_COMP_HELLINGER);
			double temp2 = compareHist(srcEdgeMatHist, objEdgeHists[j], CV_COMP_HELLINGER);

			if (temp1 > temp2)
			{
				swap(objs[i], objs[j]);
			}
		}
}

void edgeMatching_2(vector<Database>& objs, Database src)
{

	// rearrange data
	for (int i = 0; i < objs.size() - 1; i++)
		for (int j = i + 1; j < objs.size(); j++)
			if (objs[i].urlImg.compare(objs[j].urlImg) > 0)
				swap(objs[i], objs[j]);

	// calculate edge matrix for query image
	Mat srcEdgeMat = calcEdgeMat(src.srcImg);
	// and its edge histogram
	Mat srcEdgeMatHist = calcOneHist(srcEdgeMat);

	// for comparison method
	double temp1, temp2;
	for (int i = 0; i < objs.size() - 1; i++)
	{
		for (int j = i + 1; j < objs.size(); j++)
		{
			temp1 = compareHist(srcEdgeMatHist, objs[i].hists[0], CV_COMP_HELLINGER);
			temp2 = compareHist(srcEdgeMatHist, objs[j].hists[0], CV_COMP_HELLINGER);
			// if more matchable, then swap
			if (temp1 > temp2)
				swap(objs[i], objs[j]);
		}

	}

}


// as said before , this is an ulility function for calculate one histogram, especially for grayscale image case
Mat calcOneHist(Mat image)
{
	int histSize = 256;
	Mat tempHist;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	// histogram of source image's edge matrix
	calcHist(&image, 1, 0, Mat(), tempHist, 1, &histSize, &histRange, true, false);

	return tempHist;
}

//reorder Bhattacharyya distance
void bhatta_distance(vector<Database>&objs, Database src, bool gbin) 
{
	for (int i = 0; i < objs.size() - 1; i++)
		for (int j = i + 1; j < objs.size(); j++)
			if (objs[i].urlImg.compare(objs[j].urlImg) > 0)
				swap(objs[i], objs[j]);

	vector<Mat>Hists = calc_Img(src.srcImg, gbin);
	for (int i = 0; i < objs.size() - 1; i++) {
		for (int j = i + 1; j < objs.size(); j++) 
		{
			double tmp_h1 = 0;
			double tmp_h2 = 0;
			for (int ch = 0; ch < 3; ch++) 
			{
				tmp_h1 += compareHist(Hists[ch], objs[i].hists[ch],4);
				tmp_h2 += compareHist(Hists[ch], objs[j].hists[ch],4);
			}
			//swap the position of image object in an array
			//so when we want to get the samest with an srcImg
			//we just need to get elements from the start
			if (tmp_h1 > tmp_h2) {
				swap(objs[i], objs[j]);
			}
		}
	}
}

//find all files in a folder
bool find_files(vector<Database>&tmp, string img) {
	WIN32_FIND_DATAA FindFileData;

	HANDLE hFind = FindFirstFileA((img + "/*").c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return 0;
	}
	else do {
		//load data from file
		Database obj;
		obj.nameImg = FindFileData.cFileName;
		obj.urlImg = img + "/" + obj.nameImg;
		std::replace(obj.urlImg.begin(), obj.urlImg.end(), '\\', '/'); // replace all 'x' to 'y'
		obj.srcImg = imread(obj.urlImg, CV_LOAD_IMAGE_COLOR);
		//Add into database arrays
		if (!obj.srcImg.empty())
			tmp.push_back(obj);
	} while (FindNextFileA(hFind, &FindFileData));
	FindClose(hFind);
	return true;
}

//Convert image into icon
HICON convert_Icon(PTSTR imgPath) {
	HICON hIcon = NULL;
	Gdiplus::Image *image = Gdiplus::Image::FromFile(imgPath);
	Gdiplus::Bitmap* bitmap = static_cast<Gdiplus::Bitmap*>(image);

	if (bitmap)
	{
		bitmap->GetHICON(&hIcon);
		delete bitmap;
	}
	return hIcon;
}

//Convert String to LPWSTR
LPWSTR Convert_LPWSTR(string& instr) {
	int bufferlen = ::MultiByteToWideChar(CP_ACP, 0, instr.c_str(), instr.size(), NULL, 0);

	if (bufferlen == 0)
	{
		return 0;
	}
	LPWSTR widestr = new WCHAR[bufferlen + 1];
	::MultiByteToWideChar(CP_ACP, 0, instr.c_str(), instr.size(), widestr, bufferlen);
	widestr[bufferlen] = 0;
	return widestr;
}

//show result into listview
void showResult(HWND hDialog, HWND hListview, vector<Database>list, int k) {
	HIMAGELIST hImageList = ImageList_Create(GetSystemMetrics(SM_CXICON) + 125,
		GetSystemMetrics(SM_CYICON) + 90, ILC_MASK | ILC_COLOR32, 1, 1);//create a image list

	for (int i = 0; i < k; i++)
	{
		HICON hIcon = convert_Icon(Convert_LPWSTR(list[i].urlImg));//convert image to icon
		ImageList_AddIcon(hImageList, hIcon);//add icon into image list

		ListView_SetIconSpacing(hListview, 175, 150);//set the space between icon
	}
	ListView_SetImageList(hListview, hImageList, LVSIL_NORMAL);

	LVITEM listviewItem = { 0 };
	listviewItem.mask = LVIF_IMAGE | LVIF_TEXT;
	for (int i = 0; i < k; ++i)//insert each image list's element into listview
	{
		listviewItem.iItem = i;
		listviewItem.iImage = i;
		listviewItem.pszText = Convert_LPWSTR(list[i].nameImg);
		SendMessage(hListview, LVM_INSERTITEM, 0, (LPARAM)&listviewItem);
	}
}