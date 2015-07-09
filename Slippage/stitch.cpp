
#include"stdafx.h"
#include"functions.h"

#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


#include"core.h"
#include"util.h"



// Default command line args
//vector<string> img_names;
bool preview = false;
bool try_ = false;
double work_megapix = 1.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.8f;
string features_type = "orb";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "plane";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "no";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 1;

struct StitchData
{
	cv::Mat  m_warpedImg;
	cv::Mat  m_wapredAlpha;
	cv::Point m_corner;
	cv::Size  m_size;
};


int stitch_pair(const vector<Mat> &input_imgs, std::vector<StitchData> &vdata)
{
	/*imwrite("a.jpg", input_imgs[0]);
	imwrite("b.jpg", input_imgs[1]);
	imshow("a", input_imgs[0]);
	imshow("b", input_imgs[1]);
	waitKey(0);*/

	vector<CameraParams> cameras;

	//for(size_t i=0; i<_input_imgs.size(); ++i)
	//{
	//	vector<Mat> input_imgs;
	//	input_imgs.push_back(_input_imgs[0]);
	//	input_imgs.push_back(_input_imgs[i]);

    cv::setBreakOnError(true);

    // Check if have enough images
    int num_images = static_cast<int>(input_imgs.size());
    if (num_images < 2)
    {
        cout << "Need more images" << endl;
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;


    cout << "Finding features..." << endl;

    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
#if defined(HAVE_OPENCV_NONFREE) && defined(HAVE_OPENCV_)
        if (try_ && ::getCudaEnabledDeviceCount() > 0)
            finder = new SurfFeaturesFinderGpu();
        else
#endif
            finder = new SurfFeaturesFinder();
    }
    else if (features_type == "orb")
    {
        finder = new OrbFeaturesFinder();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n" << endl;
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = input_imgs[i].clone();
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            cout<< "Can't open image " << i << endl;
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        (*finder)(img, features[i]);
        features[i].img_idx = i;
        cout << "Features in image #" << i+1 << ": " << features[i].keypoints.size() << endl;

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }

    finder->collectGarbage();
    full_img.release();
    img.release();


    cout << endl <<"Pairwise matching..." << endl;

    vector<MatchesInfo> pairwise_matches;
    BestOf2NearestMatcher matcher(try_, match_conf);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();

	cout<<pairwise_matches[1].H<<endl;
	showRegist(input_imgs[0],input_imgs[1],pairwise_matches[1].H);

    // Check if we should save matches graph
    /*if (save_graph)
    {
        cout << endl << "Saving matches graph..." << endl;
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }*/

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
//    vector<string> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
 //       img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
 //   img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_subset.size());
    if(num_images < 2)
    {
        cout << "Need more images" << endl;
        return -1;
    }


	cout << endl << "Calibrating..." << endl;

    HomographyBasedEstimator estimator;
    
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        cout << "Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K() << endl;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);

	for(int i=0; i<cameras.size(); ++i)
	{
	//	cameras[i].R=cameras[0].R.inv()*cameras[i].R;
	}

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout << "Camera #" << indices[i]+1 << ":\n" << cameras[i].K() << endl;
        focals.push_back(cameras[i].focal);

		if(!(0 < cameras[i].focal && cameras[i].focal < 1e6))
		{
			cout << "bundle ajuster failed." << endl << endl;
			return -2;
		}
    }


    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }


    cout << endl << "Warping images (auxiliary)... " << endl;

    vector<Point> corners(num_images);
    vector<Mat> masks_warped(num_images);	//only this is used 
    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);

    // Preapre images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_
    if (try_ && ::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
    }
    else
#endif
    {
        if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
        else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
        else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
        else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
        else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
        else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
        else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
        else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
        else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
        else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
        else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
        else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
        else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
        else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();
    }

    if (warper_creator.empty())
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<Mat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

   
    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    compensator->feed(corners, images_warped, masks_warped);

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "voronoi")
        seam_finder = new detail::VoronoiSeamFinder();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_
        if (try_ && ::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_
        if (try_ && ::getCudaEnabledDeviceCount() > 0)
            seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
  /*  else if (seam_find_type == "dp_color")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);*/
    if (seam_finder.empty())
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
	


    cout << endl <<"Compositing..." << endl;

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

	Mat mask0;

	vdata.resize(num_images);

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        cout << endl << "Compositing image #" << indices[img_idx]+1 << endl;

        // Read image and resize it if necessary
        full_img = input_imgs[img_idx].clone();
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
        // Compensate exposure

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size());
        mask_warped = seam_mask & mask_warped;

        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f/blend_width);
                cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
            }
    //        blender->prepare(corners, sizes);
        }

			
		//if(img_idx==0)
		//{
		//	mask0=mask_warped.clone();
		//}
		//else
		//{
		//	int dx=corners[0].x-corners[img_idx].x, dy=corners[0].y-corners[img_idx].y;
		//	for(int yi=0; yi<mask_warped.rows; ++yi)
		//	{
		//		for(int xi=0; xi<mask_warped.cols; ++xi)
		//		{
		//			int x=xi-dx, y=yi-dy;

		//			if(unsigned(y)<mask0.rows&&unsigned(x)<mask0.cols && mask0.at<uchar>(y,x)!=0)
		//				mask_warped.at<uchar>(yi,xi)=0;
		//		}
		//	}
		//}

		vdata[img_idx].m_wapredAlpha=mask_warped;
		vdata[img_idx].m_warpedImg=img_warped_s;
		vdata[img_idx].m_corner=corners[img_idx];
		vdata[img_idx].m_size=img_warped_s.size();

    }

	return 0;
}


int stitch(const vector<Mat> &input_imgs, Mat &result_img, Mat &result_mask, cv::Point &refCorner)
{
	std::vector<std::vector<StitchData> > vdata(input_imgs.size());

	std::vector<Mat> ii(2);
	ii[0]=(input_imgs[0]);

	int K=0;
	for(int i=1; i<input_imgs.size(); ++i)
	{
		ii[1]=input_imgs[i];
		vdata[K].resize(2);

		if(stitch_pair(ii,vdata[K])==0)
			++K;
	}

	if(K==0)
		return -1;

	vdata.resize(K);

	refCorner=vdata[0][0].m_corner;

	std::vector<cv::Point> corners;
	std::vector<cv::Size>  sizes;

	for(int i=0; i<vdata.size(); ++i)
	{
		for(size_t j=0; j<vdata[i].size(); ++j)
		{
			vdata[i][j].m_corner+=vdata[0][0].m_corner-vdata[i][0].m_corner;
		}
	}

	corners.push_back(vdata[0][0].m_corner);
	sizes.push_back(vdata[0][0].m_size);
	for(int i=0; i<vdata.size(); ++i)
	{
		corners.push_back(vdata[i][1].m_corner);
		sizes.push_back(vdata[i][1].m_size);
	}

	 Ptr<Blender> blender;

	 if (blender.empty())
    {
        blender = Blender::createDefault(blend_type, try_);
        Size dst_sz = resultRoi(corners, sizes).size();
        float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
        if (blend_width < 1.f)
            blender = Blender::createDefault(Blender::NO, try_);
        else if (blend_type == Blender::MULTI_BAND)
        {
            MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
            mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
        }
        else if (blend_type == Blender::FEATHER)
        {
            FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
            fb->setSharpness(1.f/blend_width);
            cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
        }
    }

	 blender->prepare(corners, sizes);

	 blender->feed(vdata[0][0].m_warpedImg, vdata[0][0].m_wapredAlpha, vdata[0][0].m_corner);
	 for(int i=0; i<vdata.size(); ++i)
	 {
		 blender->feed(vdata[i][1].m_warpedImg, vdata[i][1].m_wapredAlpha, vdata[i][1].m_corner);


	 }

	 blender->blend(result_img, result_mask);

	 return 0;
}

int stitch(const cv::Mat &ref, const vector<Mat> &vsrc, Mat &result_img, Mat &result_mask, cv::Point &refCorner)
{
	std::vector<Mat> vii;
	vii.push_back(ref);
	std::copy(vsrc.begin(),vsrc.end(), std::back_inserter(vii));

	return stitch(vii,result_img,result_mask, refCorner);
}

#include"DP.h"



int refine_H(const cv::Mat &ref, const cv::Mat &src, cv::Mat &H, int nKLTFeatures, float ransacThreshold, std::vector<Point2f> *refSel, std::vector<Point2f> *srcSel)
{
	cv::Mat warpImg, warpMask;

	cv::warpPerspective(src,warpImg,H,ref.size(),cv::INTER_LINEAR,BORDER_REFLECT);

	cv::Mat mask(src.rows, src.cols, CV_8UC1);
	mask=255;

	cv::warpPerspective(mask,warpMask,H,ref.size(),cv::INTER_NEAREST);

	std::vector<Point2f> srcKeyPoints;
	getKltkeyPoints(warpImg, srcKeyPoints, nKLTFeatures, warpMask);

	std::vector<Point2f> srcMatch, refMatch;
	KLTTrackNoRansac(warpImg,ref,srcKeyPoints,srcMatch,refMatch,3);

	cv::Mat Hx=findHomographyEx(srcMatch,refMatch,CV_RANSAC, ransacThreshold, srcSel, refSel);

	H=Hx*H;

//	showMatch(src,srcSel,ref,refSel,NULL);

//	showRegist(src,ref,H,6);

	return 0;
}

#include"DP.h"

int refine_H(const cv::Mat &ref, const cv::Mat &src, const std::vector<Point2f> &srcKLT, cv::Mat &H,  float ransacThreshold, std::vector<Point2f> *refSel, std::vector<Point2f> *srcSel)
{
	cv::Mat warpImg, warpMask;

	cv::warpPerspective(src,warpImg,H,ref.size(),cv::INTER_LINEAR,BORDER_REFLECT);

	cv::Mat mask(src.rows, src.cols, CV_8UC1);
	mask=255;

	cv::warpPerspective(mask,warpMask,H,ref.size(),cv::INTER_NEAREST);

	std::vector<Point2f> srcKeyPoints(srcKLT.size());
	do_transform(H, &srcKLT[0], Point2f(0,0), &srcKeyPoints[0], int(srcKLT.size()));

	std::vector<Point2f> srcMatch, refMatch;
	KLTTrackNoRansac(warpImg,ref,srcKeyPoints,srcMatch,refMatch,3);

	cv::Mat Hx=findHomographyEx(srcMatch,refMatch,CV_RANSAC, ransacThreshold, srcSel, refSel);

	H=Hx*H;

	return 0;
}

void _update_stitched(cv::Mat &dest, cv::Mat &conf, cv::Mat &index, int k, cv::Point origin, const cv::Mat &src, const cv::Mat &mask, float src_conf)
{
	for(int yi=0; yi<src.rows; ++yi)
	{
		int dy=origin.y+yi;

		if(uint(dy)<dest.rows)
		{
			for(int xi=0; xi<src.cols; ++xi)
			{
				int dx=origin.x+xi;
				if(uint(dx)<dest.cols && mask.at<uchar>(yi,xi)!=0)
				{
					if(src_conf>conf.at<float>(dy,dx))
					{
						conf.at<float>(dy,dx)=src_conf;
						memcpy(dest.ptr(dy,dx), src.ptr(yi,xi), 3);
						index.at<ushort>(dy,dx)=(ushort)k;
					}
				}
			}
		}
	}
}

int stitch(const cv::Mat &ref, const vector<Mat> &vsrc, vector<Mat> &vH, Mat &result_img, Mat &result_index, cv::Point &refCorner)
{
	std::vector<Point2f> refPoints,srcPoints;
//	std::vector<float>  vconf(vH.size(),0);
	std::vector<float>  vconf;

	for(size_t i=0; i<vH.size(); ++i)
	{
		cv::Mat HT(vH[i].clone());
		refine_H(ref,vsrc[i],HT,300,3.0f,&refPoints,&srcPoints);
		
		vH[i]=HT;

		float error=_get_transform_error(HT,srcPoints,refPoints);
		vconf.push_back(1000/(error+1e-3));
	}

	std::vector<cv::Point2f>  vcorners;

	cv::Size bgSize=ref.size();
	cv::Point2f  bgCorners[]={Point2f(0,0),Point2f(0,bgSize.height),Point2f(bgSize.width,bgSize.height),Point2f(bgSize.width,0)};

	for(int i=0; i<4; ++i)
		vcorners.push_back(bgCorners[i]);

	std::vector<cv::Point2f> vtl;

	for(size_t k=0; k<vsrc.size(); ++k)
	{
		bgSize=vsrc[k].size();
		cv::Point2f icorners[]={Point2f(0,0),Point2f(0,bgSize.height),Point2f(bgSize.width,bgSize.height),Point2f(bgSize.width,0)};
		cv::Point2f icornersWarped[4];

		do_transform(cv::Matx33d(vH[k]), icorners, Point2f(0,0), icornersWarped, 4); 

	//	float conf=1000/(1e-3+_get_bg_distortion(icornersWarped));
	//	vconf[k]=conf;

		vtl.push_back(icornersWarped[0]);
		for(int i=0; i<4; ++i)
			vcorners.push_back(icornersWarped[i]);
	}

	cv::Rect bb(_get_bounding_box(vcorners));
	result_img=cv::Mat(bb.height,bb.width,CV_8UC3);
	result_img=0;
	refCorner=cv::Point(-bb.x,-bb.y);

	printf("\nbb=%d,%d,%d,%d",bb.x,bb.y,bb.width,bb.height);

	result_index=cv::Mat(bb.height,bb.width,CV_16U);
	result_index=-1;

	cv::Mat conf(bb.height,bb.width,CV_32FC1);
	conf=0;

	const float MAX_CONF=1e8;

	cv::Mat mask(ref.rows,ref.cols, CV_8UC1);
	mask=255;
	_update_stitched(result_img, conf, result_index,0, cv::Point(-bb.x,-bb.y), ref, mask, MAX_CONF);

	cv::Mat T( (cv::Mat_<double>(3,3)<<1,0,-bb.x, 0,1,-bb.y, 0,0,1) );

	for(size_t k=0; k<vsrc.size(); ++k)
	{
		cv::Mat warped, warped_mask;
		cv::Mat H(T*vH[k]);

		cv::warpPerspective(vsrc[k], warped, H, result_img.size(),INTER_LINEAR,BORDER_REFLECT);

		mask=cv::Mat(vsrc[k].rows, vsrc[k].cols, CV_8UC1);
		mask=255;

		cv::warpPerspective(mask, warped_mask, H, result_img.size(), INTER_NEAREST);
		_update_stitched(result_img,conf, result_index, k+1, cv::Point(0,0), warped, warped_mask, vconf[k]);
	}

	imshow("stitched",result_img);
//	waitKey();

	return 0;
}



void stitch_test()
{
	std::vector<cv::Mat> vimg;

	for(int i=1; i<=3; ++i)
	{
	//	if(i==1||i==2)
		vimg.push_back(cv::imread(ff::StrFormat("..\\03 (%d).jpg",i)));
	}

	cv::Mat result, result_mask;

//	stitch(vimg,result,result_mask);

	imwrite("..\\pano.jpg",result);
}

void get_stitch_index(const Mat &index_img, std::vector<int> &vindex, int maxIndex=1024)
{
	std::vector<char> mask(maxIndex,0);

	for(int yi=0; yi<index_img.rows; ++yi)
	{
		for(int xi=0; xi<index_img.cols; ++xi)
		{
			int idx=index_img.at<short>(yi,xi)-1;
			if(idx>=0 && mask[idx]==0)
			{
				vindex.push_back(idx);
				mask[idx]=1;
			}
		}
	}
}

struct StitchBgData
{
	cv::Mat  m_H;
	cv::Mat  m_img;
	int      m_index;
};

void add_prev_nbrs(const std::vector<StitchBgData> &lastNbr, const std::vector<int> &vindex, std::vector<StitchBgData> &curNbr)
{
	for(size_t i=0; i<vindex.size(); ++i)
	{
		int idx=vindex[i];
		if(uint(idx)<lastNbr.size())
		{
			size_t j=0;
			for(; j<curNbr.size(); ++j)
			{
				if(curNbr[j].m_index==lastNbr[ idx ].m_index)
					break;
			}

			if(j==curNbr.size())
			{
				curNbr.push_back(lastNbr[idx]);
			}

		}
	}
}

#if 0

int CSlippage::Synthesis()
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	VideoWriter new_video;

	for(uint i=30; i<fgDataSet.Size(); ++i)
	{
		std::cout << "frame:" << i <<endl;
		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);

		int correspond_bg_index = fg_fd->m_Correspondence;
		Mat correspond_bg_affine = fg_fd->m_CorrespondenceAffine;

		int videoId = bgDataSet.GetAtIndex(correspond_bg_index)->m_videoID;
		int frameId = bgDataSet.GetAtIndex(correspond_bg_index)->m_FrameID;
		ImageSetReader&bgIsr(*bgIsrList[videoId]);

		fgIsrPtr->SetPos(i);
		alphaIsrPtr->SetPos(i);
		bgIsr.SetPos(frameId);
		
		Mat *fg_img = fgIsrPtr->Read();
		Mat *fg_alpha = alphaIsrPtr->Read();
		Mat _bg_img = bgIsr.Read()->clone(),*bg_img=&_bg_img;
		Mat bg_img_warp;

		bool stitched=false;

		if(fabs(fg_fd->m_Errors.m_coverage)<1)
			cv::warpPerspective(*bg_img,bg_img_warp,correspond_bg_affine,fg_img->size());
		else
		{
			Mat stitched_bg, stitched_bg_mask;

			std::vector<cv::Mat> nbr;
			BgFrameData *bd=bgDataSet.GetAtIndex(fg_fd->m_Correspondence);
			for(size_t j=0; j<bd->m_nbr.size(); ++j)
			{
				BgFrameData *bj=bgDataSet.GetAtIndex(bd->m_nbr[j].m_index);
				if(bj->m_videoID!=bd->m_videoID||abs(bj->m_FrameID-bd->m_FrameID)>3)
				{
					cv::Mat *ij=bgIsrList[bj->m_videoID]->Read(bj->m_FrameID);
					nbr.push_back(ij->clone());

		//			showRegist(*bg_img,*ij,bd->m_nbr[j].m_T);

					break;
				}
			}

		/*	imshow("bg",*bg_img);
			imshow("nbr",nbr.front());
			cv::waitKey(0);*/

			cv::Point refCorner;

			if(stitch(*bg_img, nbr, stitched_bg, stitched_bg_mask, refCorner)==0)
			{
				cv::Mat bgT( (cv::Mat_<double>(3,3)<<1,0,-refCorner.x, 0,1,-refCorner.y, 0,0,1) );
				bgT=correspond_bg_affine*bgT;

				cv::warpPerspective(stitched_bg,bg_img_warp,bgT,fg_img->size());

				printf("\nframe %d",i);
				stitched=true;
			//	system("pause");
			}
			else
				cv::warpPerspective(*bg_img,bg_img_warp,correspond_bg_affine,fg_img->size());
		}

		Mat new_img(fg_img->rows, fg_img->cols, CV_8UC3);

		ff::for_each_pixel_3_1(_CV_DWHSC(*fg_img),_CV_DSC(*fg_alpha),_CV_DSC(bg_img_warp),_CV_DSC(new_img),ff::iop_alpha_blend_i8u<3>());

		if( i == 0 )
		{
			new_video = VideoWriter(dataDir+"result.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
		}

		new_video.write(new_img);

		cv::imshow("new video", new_img);
		cv::waitKey(stitched? 0 : 10);
	}
}

//#else

int CSlippage::Synthesis(bool bStitch)
{
	FgDataSet fgDataSet;
	BgDataSet bgDataSet;

	if(this->_LoadDataSet(&fgDataSet, &bgDataSet)<0)
		return -1;

	VideoWriter new_video;

	int lastCorrespondence=-1;
	Mat lastStitched;
	Point2f lastRefCorner;
	std::vector<StitchBgData> lastNbr;
	Mat   lastIndex;

	cv::Size fgSize=cv::Size(fgIsrPtr->Width(),fgIsrPtr->Height());
	cv::Point2f fgCorners[4]={Point2f(0.0f,0.0f), Point2f(0,(float)fgSize.height), Point2f((float)fgSize.width,(float)fgSize.height), Point2f((float)fgSize.width,0) };

	cv::Size bgSize=cv::Size(bgIsrList.front()->Width(), bgIsrList.front()->Height());
	cv::Point2f bgCorners[4]={Point2f(0.0f,0.0f), Point2f(0,(float)bgSize.height), Point2f((float)bgSize.width,(float)bgSize.height), Point2f((float)bgSize.width,0) };
	cv::Point2f bgCornersWarped[4];


	for(uint i=0; i<fgDataSet.Size(); ++i)
	{
		FgFrameData *fg_fd = fgDataSet.GetAtIndex(i);

		int correspond_bg_index = fg_fd->m_Correspondence;
		Mat correspond_bg_affine = fg_fd->m_CorrespondenceAffine;

		int videoId = bgDataSet.GetAtIndex(correspond_bg_index)->m_videoID;
		int frameId = bgDataSet.GetAtIndex(correspond_bg_index)->m_FrameID;
		ImageSetReader&bgIsr(*bgIsrList[videoId]);

		fgIsrPtr->SetPos(i);
		alphaIsrPtr->SetPos(i);
		bgIsr.SetPos(frameId);
		
		Mat *fg_img = fgIsrPtr->Read();
		Mat *fg_alpha = alphaIsrPtr->Read();
		Mat _bg_img = bgIsr.Read()->clone(),*bg_img=&_bg_img;
		Mat bg_img_warp;

		bool stitched=false;

		cv::warpPerspective(*bg_img,bg_img_warp,correspond_bg_affine,fg_img->size());
		imshow("no stitch", bg_img_warp);

		std::cout << "frame:" << i <<"c="<<fg_fd->m_Errors.m_coverage<<endl;

		do_transform(correspond_bg_affine,bgCorners,cv::Point2f(0,0),bgCornersWarped,4);
		float coverage=_get_coverage(bgCornersWarped,fgCorners,fgSize.width*fgSize.height);

		if(!bStitch||fabs(coverage)<1)
			cv::warpPerspective(*bg_img,bg_img_warp,correspond_bg_affine,fg_img->size());
		else
		{
			Mat stitched_bg, stitched_bg_index;
			cv::Point refCorner;

			if(lastCorrespondence==correspond_bg_index)
			{
				stitched_bg=lastStitched;
				refCorner=lastRefCorner;
			}
			else
			{
				cv::Mat ScaleT( (cv::Mat_<double>(3,3)<<m_bgWorkScale,0,0, 0,m_bgWorkScale,0, 0,0,1) );

				std::vector<StitchBgData>  bgCand;
				BgFrameData *bd=bgDataSet.GetAtIndex(fg_fd->m_Correspondence);
				for(size_t j=0; j<bd->m_nbr.size(); ++j)
				{
					BgFrameData *bj=bgDataSet.GetAtIndex(bd->m_nbr[j].m_index);
					StitchBgData bgd;
				//	if(bj->m_videoID!=bd->m_videoID)
				//	if(bd->m_nbr[j].m_distortion_err<0.5)
				//	if(bd->m_nbr[j].m_flag&BgFrameData::NF_STITCH_CAND) // && bj->m_videoID==bd->m_videoID)
					{
						cv::Mat *ij=bgIsrList[bj->m_videoID]->Read(bj->m_FrameID);
						
						bgd.m_img=ij->clone();
						bgd.m_H=(ScaleT.inv()*bd->m_nbr[j].m_T*ScaleT).inv();
						bgd.m_index=bd->m_nbr[j].m_index;

						bgCand.push_back(bgd);
					}
				}

				if(lastIndex.data && !lastNbr.empty())
				{
					std::vector<int> vindex;
					get_stitch_index(lastIndex,vindex);
			//		add_prev_nbrs(lastNbr,vindex,bgCand);
				}

				std::vector<cv::Mat> nbr, vH;
				for(size_t j=0; j<bgCand.size(); ++j)
				{
					nbr.push_back(bgCand[j].m_img);
					vH.push_back(bgCand[j].m_H.clone());
				}

				stitch(*bg_img, nbr, vH, stitched_bg, stitched_bg_index, refCorner);

				for(size_t j=0; j<bgCand.size(); ++j)
				{
					bgCand[j].m_H=vH[j];
				}
				
				lastCorrespondence=correspond_bg_index;
				lastStitched=stitched_bg;
				lastRefCorner=refCorner;

				lastNbr.swap(bgCand);
			}

			if(stitched_bg.data)
			{
				cv::Mat bgT( (cv::Mat_<double>(3,3)<<1,0,-refCorner.x, 0,1,-refCorner.y, 0,0,1) );
				bgT=correspond_bg_affine*bgT;

				cv::warpPerspective(stitched_bg,bg_img_warp,bgT,fg_img->size());

				if(stitched_bg_index.data)
				{
					cv::warpPerspective(stitched_bg_index,lastIndex,bgT,fg_img->size(),INTER_NEAREST);
				}

				stitched=true;
			}
			else
				cv::warpPerspective(*bg_img,bg_img_warp,correspond_bg_affine,fg_img->size());
		}

		Mat new_img(fg_img->rows, fg_img->cols, CV_8UC3);

		ff::for_each_pixel_3_1(_CV_DWHSC(*fg_img),_CV_DSC(*fg_alpha),_CV_DSC(bg_img_warp),_CV_DSC(new_img),ff::iop_alpha_blend_i8u<3>());

		if( !new_video.isOpened() )
		{
			new_video = VideoWriter(dataDir+"result.avi", CV_FOURCC('D','I','V','X'), 25, Size(new_img.cols, new_img.rows));
		}

		new_video.write(new_img);

		cv::imshow("new video", new_img);
	//	cv::waitKey(stitched? 0 : 10);
		cv::waitKey(10);
	}

	return 0;
}


#endif

