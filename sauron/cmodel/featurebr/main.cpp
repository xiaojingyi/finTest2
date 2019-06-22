/**
 * FileName:       main.cpp
 * Type:           C++
 * Encoding:       utf-8
 * Copyright:      (c) 2010 Jingyi Xiao
 * Note:           This source file is NOT a freeware
 * Authors:        Jingyi Xiao <kxwarning@126.com>
 * Description:
 */

#include <iostream>

using namespace std;

#include <openbr/openbr_plugin.h>
static void printTemplate(const br::Template &t) {
    cv::Mat a = t.m();
//    cout << a.rows << ' ' << a.cols << '\n';
    const int size = a.rows * a.cols;
    double sum=0;
    for(int i = 0; i < size; i++) {
//        sum += (int) a.data[i];
        cout << (int) a.data[i] << ',';
/*        cout << (int)a.data[i] << ',';*/
    }
    cout << '\n';
//    cout << a << '\n';
//    cout << sum << '\n';
}

float compare(cv::Mat &a, cv::Mat &b){
    const uchar *aData = a.data;
    const uchar *bData = b.data;
    const int size = a.rows * a.cols;
    float likelihood = 0;
    int distance = 0;
    for (int i=0; i<size; i++){
//        likelihood += (256 - (abs(aData[i]-bData[i]))) / ((float)256 * size);
//        likelihood += 1 - pow(0.8, abs(aData[i]-bData[i]));
        distance = (int)aData[i] - (int)bData[i];
        likelihood += distance * distance;
    }
    likelihood = sqrt(likelihood);
    return likelihood;
}

int main(int argc, char *argv[])
{
    br::Context::initialize(argc, argv);
    // Retrieve classes for enrolling and comparing templates using the FaceRecognition algorithm
    QSharedPointer<br::Transform> transform = br::Transform::fromAlgorithm(argv[1]);
//    QSharedPointer<br::Distance> distance = br::Distance::fromAlgorithm("FaceRecognition");
    // Initialize templates
    br::Template queryA(argv[2]);
//    br::Template target(argv[2]);
    // Enroll templates
    queryA >> *transform;
    printTemplate(queryA);
/*    target >> *transform;
    printTemplate(target);
    double res = cv::norm(queryA.m()-target.m());
    cout << "distance: " << res << '\n';
    res = cv::norm(queryA.m()-queryA.m());
    cout << "distance: " << res << '\n';
    res = compare(queryA.m(),target.m());
    cout << "distance: " << res << '\n';
    res = compare(queryA.m(),queryA.m());
    cout << "distance: " << res << '\n';
    // Compare templates
    float comparisonA = distance->compare(target, queryA);
    // Scores range from 0 to 1 and represent match probability
    printf("match score: %.3f\n", comparisonA);*/
    br::Context::finalize();
    return 0;
}


