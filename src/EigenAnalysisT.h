//  Created by Xiaoyong Bai on 11/09/16.
//
//  The class is to do the eigenvalue analysis for BEM.

#ifndef EigenAnalysisT_hpp
#define EigenAnalysisT_hpp

#include <iostream>

#include "slepcsys.h"
#include "slepceps.h"
#include "petscmat.h"


namespace Stability{
    
class EigenAnalysisT
{
public:
    
    EigenAnalysisT();
    ~EigenAnalysisT();
    
    //set number and size of H matrices
    void SetMatrixNumSize(int num, int size);
    
    //set matrices
    void SetMatrixSystem_Direct(double*);
    
    void SetMatrixSystem_Ave(double* H, double a1, double a2);

    //form amplification matrix
    void FormA_Direct(void);
    void FormA_Ave(void);
    
    //compute the largest eigenvalue
    double LargestEigen_Direct(void);
    double LargestEigen_Ave(void);

    
    //Compute the Inverse of a matrix
    void InverseMPI(Mat A, Mat& A_Inv);
    
    //C=A*B
    void DenseMatMult(Mat A, Mat B, Mat& C);
    
    
    //Shell for A
    static void MatMult_A_Shell(Mat A, Vec x, Vec y);
    
    //Decompose V into m sub vecotors with the same length
    void DecomposeVector(Vec V, int m, Vec* sub_V);
    void CombineVector(int m, Vec* sub_V, Vec V);
    
    int GetNumMatrix();
    Mat* Get_H_Ave();
    
private:
    int fNumMatrix; //Number of matrices to be used for stability analysis.
    int fNumRow; //Number of rows in each matrix
    
    int fHFirst, fHLast;
    int fAFirst_direct, fALast_direct;
    int fAFirst_ave, fALast_ave;
    
    int fRank;
    
    Mat fH_Inv_Direct;
    Mat fH_Inv_Ave;
    
    Mat* fH_Direct;
    Mat* fH_Ave;
    
    //Amplification matrix for direct stepping and averaging stepping
    Mat fA_Direct;
    Mat fA_Ave;
    
    
    double fa1, fa2, fa3;//alpha_1, alpha_2 are two parameters in averaging method.
    
    PetscErrorCode fIerr;

};//end of definition of class EigenAnalysisTT
    


    
}//end of namespace



#endif /* EigenAnalysisT_hpp */
