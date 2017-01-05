//  Created by Xiaoyong Bai on 11/09/16.
//
//  The class is to read G and H matrices.

#ifndef GHReaderT_hpp
#define GHReaderT_hpp

#include <iostream>

namespace Stability{
    
class GHReaderT
{
public:
    
    GHReaderT();
    ~GHReaderT();

    void ReadMatrixNumber();
    void ReadMatrixNumRow();
    
    //Read all the G, H matrices.
    void ReadMatrices();
    
    //Exchange the Columns of G and H, based on the displacement boundary condition
    void ExchangeColumn_GH();
    
    //Read one G or H with the name File.
    void ReadMatrix(double* H, int firstRow, int lastRow, const char* File);
    
    //Read the first 3 H matrices, i.e. H1, H2, H3
    void GetH(double** H);
    
    int GetMatrixNumber();
    int GetMatrixNumRow();
    
    void GetMatrices(double** H1, double** H2, double** G1, double** G2);
    

private:
    int fNumMatrix; //Number of matrices to be used for stability analysis.
    
    int fNumRow; //Number of rows in each G and H matrix
    
    //When linear interploation is adopted for time discretization, two H and two G matrices will be generated each step.
    //Store all the matrices in one array.
    //All the matrices are of the same size, fNumRow*fNumRow.
    double* fH1;
    double* fH2;
    double* fG1;
    double* fG2;
    
    double* fH;
    double* fG;
    
};//end of definition of class GHReaderT
    


    
}//end of namespace



#endif /* GHReaderT_hpp */
