// GHReaderT.cpp
//
//  Created by Xiaoyong Bai on 11/08/16.


#include "EigenAnalysisT.h"

#include <iostream>
#include <fstream>
#include "sstream"
#include <cstring>
#include "iomanip"

#include "slepcsys.h"
#include "cmath"
#include "petscmat.h"
#include "petscksp.h"

#include <ctime>

using namespace Stability;
using namespace std;


EigenAnalysisT::EigenAnalysisT()
{
    fNumMatrix=0;
    fNumRow=0;
    
    fH_Direct=PETSC_NULL;
    fH_Ave=PETSC_NULL;
    
    fA_Direct=PETSC_NULL;
    fA_Ave=PETSC_NULL;
    
    fa1=0.0;
    fa2=0.0;
    fa3=1-fa1-fa2;
    
    MPI_Comm_rank(PETSC_COMM_WORLD, &fRank );
}


EigenAnalysisT::~EigenAnalysisT()
{
    MatDestroy(&fH_Inv_Direct);
    MatDestroy(&fH_Inv_Ave);
    
    if (fH_Direct) delete [] fH_Direct;
    if (fH_Ave) delete [] fH_Ave;
    
    MatDestroy(&fA_Direct);
    MatDestroy(&fA_Ave);
}


void EigenAnalysisT::SetMatrixNumSize(int num, int size)
{
    fNumMatrix=num;
    fNumRow=size;
    
    //Allocate matrices
    fH_Direct=new Mat[fNumMatrix];
        
    for (int i=0; i<fNumMatrix; i++) {
        fIerr=MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, fH_Direct+i);
    }
        
    fH_Ave=new Mat[fNumMatrix+1];
    for (int i=0; i<=fNumMatrix; i++) {
        fIerr=MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, fH_Ave+i);
    }
    
    MatGetOwnershipRange(fH_Direct[0], &fHFirst, &fHLast);
    fHLast -= 1;
}


void EigenAnalysisT::SetMatrixSystem_Direct(double* H)
{
    int loc_num_row=fHLast-fHFirst+1;
    int single_size=loc_num_row*fNumRow;
    
    int *row_index, *column_index;
    row_index=new int[loc_num_row];
    column_index=new int[fNumRow];
    
    for (int ri=0; ri<loc_num_row; ri++) {
        row_index[ri]=fHFirst+ri;
    }
    for (int ci=0; ci<fNumRow; ci++) {
        column_index[ci]=ci;
    }
    
    //Extract H0, i.e., the first matrix in H1
    Mat H0;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H0);
    
    MatSetValues(H0, loc_num_row, row_index, fNumRow, column_index, H, INSERT_VALUES);
    MatAssemblyBegin(H0, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H0, MAT_FINAL_ASSEMBLY);
        
    //Compute inverse of H0.
    Mat H0_Inv;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H0_Inv);
    InverseMPI(H0, H0_Inv);
    MatScale(H0_Inv, -1);
    
    //cout <<">>>>>>>>>>>"<<endl;
    //cout <<"H0 is"<<endl;
    //MatView(H0, PETSC_VIEWER_STDOUT_WORLD);
    
    //cout <<">>>>>>>>>>>"<<endl;
    //cout <<"Inverse of H0 is"<<endl;
    //MatView(H0_Inv, PETSC_VIEWER_STDOUT_WORLD);
    
    Mat H_temp;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H_temp);

    Mat H_temp_1;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H_temp_1);
    MatZeroEntries(H_temp_1);
    
    for (int i=0; i<fNumMatrix; i++) {
        MatSetValues(H_temp, loc_num_row, row_index, fNumRow, column_index, H+(i+1)*single_size, INSERT_VALUES);
        MatAssemblyBegin(H_temp, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H_temp, MAT_FINAL_ASSEMBLY);
        
        //cout <<"H_temp " << i << " is "<<endl;
        //MatView(H_temp, PETSC_VIEWER_STDOUT_WORLD);
        
        DenseMatMult(H0_Inv, H_temp, fH_Direct[i]);
        
        //MatMatMult(H0_Inv, H_temp, MAT_REUSE_MATRIX, PETSC_DEFAULT, fH_Direct+i);
        //cout <<"H_Direct " << i << " is "<<endl;
        //MatView(fH_Direct[i], PETSC_VIEWER_STDOUT_WORLD);
    }
    
    delete [] column_index;
    delete [] row_index;
    
    MatDestroy(&H0);
    MatDestroy(&H0_Inv);
    MatDestroy(&H_temp);
}



void EigenAnalysisT::SetMatrixSystem_Ave(double *H, double a1, double a2)
{
    fa1=a1;
    fa2=a2;
    fa3=1.0-fa1-fa2;
    
    int loc_num_row=fHLast-fHFirst+1;
    int single_size=loc_num_row*fNumRow;
    
    int *row_index, *column_index;
    row_index=new int[loc_num_row];
    column_index=new int[fNumRow];
    
    for (int ri=0; ri<loc_num_row; ri++) {
        row_index[ri]=fHFirst+ri;
    }
    for (int ci=0; ci<fNumRow; ci++) {
        column_index[ci]=ci;
    }
    
    //Generate H0 and its inverse
    Mat H0, H_temp_1, H_temp_2, H_temp_3;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H0);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H_temp_1);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H_temp_2);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H_temp_3);
    
    
    MatSetValues(H0, loc_num_row, row_index, fNumRow, column_index, H, INSERT_VALUES);
    MatAssemblyBegin(H0, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H0, MAT_FINAL_ASSEMBLY);
    MatScale(H0, 2.0*fa1+fa3);
    
    MatSetValues(H_temp_1, loc_num_row, row_index, fNumRow, column_index, H+single_size, INSERT_VALUES);
    MatAssemblyBegin(H0, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H0, MAT_FINAL_ASSEMBLY);
    
    MatAXPY(H0, fa1, H_temp_1, SAME_NONZERO_PATTERN);
    
    //Compute inverse of H0.
    Mat H0_Inv;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &H0_Inv);
    InverseMPI(H0, H0_Inv);
    MatScale(H0_Inv, -1);
    
    std::time_t  result = std::time(NULL);
    std::cout << std::asctime(std::localtime(&result)) <<endl;
    
    //cout <<">>>>>>>>>>>"<<endl;
    //cout <<"H0 is"<<endl;
    //MatView(H0, PETSC_VIEWER_STDOUT_WORLD);
    
    //cout <<">>>>>>>>>>>"<<endl;
    //cout <<"Inverse of H0 is"<<endl;
    //MatView(H0_Inv, PETSC_VIEWER_STDOUT_WORLD);
        
    //Form H matrices for averaging method
    MatSetValues(H_temp_1, loc_num_row, row_index, fNumRow, column_index, H+0*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
        
    MatSetValues(H_temp_2, loc_num_row, row_index, fNumRow, column_index, H+1*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
        
    MatSetValues(H_temp_3, loc_num_row, row_index, fNumRow, column_index, H+2*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
        
    MatScale(H_temp_1, fa2-fa1);
    MatAXPY(H_temp_1, fa3, H_temp_2, SAME_NONZERO_PATTERN);
    MatAXPY(H_temp_1, fa1, H_temp_3, SAME_NONZERO_PATTERN);
        
    DenseMatMult(H0_Inv, H_temp_1, fH_Ave[0]);
    //MatMatMult(H0_Inv, H_temp_1, MAT_REUSE_MATRIX, PETSC_DEFAULT, fH_Ave);
    
    for (int i=1; i<fNumMatrix-1; i++) {
        MatSetValues(H_temp_1, loc_num_row, row_index, fNumRow, column_index, H+(i+0)*single_size, INSERT_VALUES);
        MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
            
        MatSetValues(H_temp_2, loc_num_row, row_index, fNumRow, column_index, H+(i+1)*single_size, INSERT_VALUES);
        MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
            
        MatSetValues(H_temp_3, loc_num_row, row_index, fNumRow, column_index, H+(i+2)*single_size, INSERT_VALUES);
        MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
            
        MatScale(H_temp_1, fa2);
        MatAXPY(H_temp_1, fa3, H_temp_2, SAME_NONZERO_PATTERN);
        MatAXPY(H_temp_1, fa1, H_temp_3, SAME_NONZERO_PATTERN);
            
        DenseMatMult(H0_Inv, H_temp_1, fH_Ave[i]);
    }
        
    //compute the second to the last
    MatSetValues(H_temp_1, loc_num_row, row_index, fNumRow, column_index, H+(fNumMatrix-1)*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
        
    MatSetValues(H_temp_2, loc_num_row, row_index, fNumRow, column_index, H+(fNumMatrix)*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
        
    MatScale(H_temp_1, fa2);
    MatAXPY(H_temp_1, fa3, H_temp_2, SAME_NONZERO_PATTERN);
    
    DenseMatMult(H0_Inv, H_temp_1, fH_Ave[fNumMatrix-1]);
        
    //compute the last 1
    MatSetValues(H_temp_1, loc_num_row, row_index, fNumRow, column_index, H+fNumMatrix*single_size, INSERT_VALUES);
    MatAssemblyBegin(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(H_temp_1, MAT_FINAL_ASSEMBLY);
    MatScale(H_temp_1, fa2);

    DenseMatMult(H0_Inv, H_temp_1, fH_Ave[fNumMatrix]);
    
    delete [] row_index;
    delete [] column_index;
    
    
    //Transpose all the matrices
    for (int i=0; i<fNumMatrix+1; i++) {
        MatTranspose(fH_Ave[i], MAT_REUSE_MATRIX, fH_Ave+i);
    }
    
    MatDestroy(&H0);
    MatDestroy(&H0_Inv);
    MatDestroy(&H_temp_1);
    MatDestroy(&H_temp_2);
    MatDestroy(&H_temp_3);
}



void EigenAnalysisT::FormA_Direct()
{
    int NumARow=fNumMatrix*fNumRow;
    
    //Allocate A
    if (!fA_Direct) {
        MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NumARow, NumARow, PETSC_NULL, &fA_Direct);
    }
    MatZeroEntries(fA_Direct);
    MatAssemblyBegin(fA_Direct, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(fA_Direct, MAT_FINAL_ASSEMBLY);
    
    
    //Insert the blocks in the 1st line
    int numRow=fHLast-fHFirst+1;
    int numColumn=fNumRow;
    double* vec=new double[numRow*numColumn];
    
    int* row=new int[numRow];
    int* column=new int[numColumn];
    int* column_global=new int[numColumn];
    for (int i=fHFirst; i<=fHLast; i++) {
        row[i-fHFirst]=i;
    }
    for (int i=0; i<numColumn; i++) {
        column[i]=i;
    }
    
    for (int hi=0; hi<fNumMatrix; hi++) {
        MatGetValues(fH_Direct[hi], numRow, row, numColumn, column, vec);
        
        for (int ci=0; ci<numColumn; ci++) {
            column_global[ci]=ci+hi*numColumn;
        }
        
        MatSetValues(fA_Direct, numRow, row, numColumn, column_global, vec, INSERT_VALUES);
        MatAssemblyBegin(fA_Direct, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(fA_Direct, MAT_FINAL_ASSEMBLY);
    }
    
    //Get indices between rank
    int size, rank;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    MatGetOwnershipRange(fA_Direct, &fAFirst_direct, &fALast_direct);
    
    fALast_direct -= 1;
    //cout <<   "Rank=" << rank << " first row="<<first_row << " last row=" <<last_row;
    
    //insert 1 in the other blocks
    if (fAFirst_direct <= fNumRow-1) {
        if (fALast_direct > fNumRow-1){
            for (int ri=fNumRow; ri<=fALast_direct; ri++) {
                MatSetValue(fA_Direct, ri, ri-fNumRow, 1.0, INSERT_VALUES);
            }
        }
    }else{
        for (int ri=fAFirst_direct; ri<=fALast_direct; ri++) {
            MatSetValue(fA_Direct, ri, ri-fNumRow, 1.0, INSERT_VALUES);
        }
    }
    
    MatAssemblyBegin(fA_Direct, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(fA_Direct, MAT_FINAL_ASSEMBLY);
    
    delete [] row;
    delete [] column;
    delete [] column_global;
    delete [] vec;
    
    //cout <<"fA_Direct is"<<endl;
    //MatView(fA_Direct,PETSC_VIEWER_STDOUT_WORLD);
}


void EigenAnalysisT::FormA_Ave()
{
    int NumARow=(fNumMatrix+1)*fNumRow;
    
    //Allocate A
    if (!fA_Ave) {
        //MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NumARow, NumARow, PETSC_NULL, &fA_Ave);
        MatCreate(PETSC_COMM_WORLD, &fA_Ave);
        MatSetSizes(fA_Ave, PETSC_DECIDE, PETSC_DECIDE, NumARow, NumARow);
        MatSetType(fA_Ave, MATMPIAIJ);
        
        //PreAllocate A
        Vec v_test;
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, NumARow, &v_test);
        
        int low, high;
        VecGetOwnershipRange(v_test, &low, &high);
        high=high-1;
        
        int d_nz, o_nz;
        if (low<=fNumRow-1) {
            if (high<=fNumRow-1) {
                d_nz=high-low+1;
            }else{
                d_nz=fNumRow-low;
            }
        }else{
            d_nz=0;
        }
        o_nz=fNumRow-d_nz;
        
        MatMPIAIJSetPreallocation(fA_Ave, d_nz, PETSC_NULL, o_nz, PETSC_NULL);
        MatSetOption(fA_Ave, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        //cout << "d_nz="<< d_nz << " o_nz="<<o_nz<<" A is created \n";
    }
    
    MatZeroEntries(fA_Ave);
    MatAssemblyBegin(fA_Ave, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(fA_Ave, MAT_FINAL_ASSEMBLY);
    
    //Insert the blocks in the 1st line
    int numRow=fHLast-fHFirst+1;
    int numColumn=fNumRow;
    double* vec=new double[numRow*numColumn];
    
    int* row=new int[numRow];
    int* column=new int[numColumn];
    int* row_global=new int[numRow];
    for (int i=fHFirst; i<=fHLast; i++) {
        row[i-fHFirst]=i;
    }
    for (int i=0; i<numColumn; i++) {
        column[i]=i;
    }
    
    for (int hi=0; hi<fNumMatrix+1; hi++) {
        MatGetValues(fH_Ave[hi], numRow, row, numColumn, column, vec);
        
        for (int ri=0; ri<numRow; ri++) {
            row_global[ri]=ri+hi*fNumRow+fHFirst;
        }
        
        MatSetValues(fA_Ave, numRow, row_global, numColumn, column, vec, INSERT_VALUES);
        MatAssemblyBegin(fA_Ave, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(fA_Ave, MAT_FINAL_ASSEMBLY);
    }
    
    //Get indices between rank
    int size, rank;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    MatGetOwnershipRange(fA_Ave, &fAFirst_ave, &fALast_ave);
    
    fALast_ave -= 1;
    
    //insert 1 in the other blocks
    if (fAFirst_ave <= fNumRow-1) {
        if (fALast_ave > fNumRow-1){
            for (int ri=fNumRow; ri<=fALast_ave; ri++) {
                MatSetValue(fA_Ave, ri-fNumRow, ri, 1.0, INSERT_VALUES);
            }
        }
    }else{
        for (int ri=fAFirst_ave; ri<=fALast_ave; ri++) {
            MatSetValue(fA_Ave, ri-fNumRow, ri, 1.0, INSERT_VALUES);
        }
    }
    
    MatAssemblyBegin(fA_Ave, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(fA_Ave, MAT_FINAL_ASSEMBLY);
    
    //MatView(fA_Ave,PETSC_VIEWER_STDOUT_WORLD);
    
    delete [] column;
    delete [] row;
    delete [] row_global;
    delete [] vec;
}



double EigenAnalysisT::LargestEigen_Direct(void)
{
    FormA_Direct();
    
    //Create the eigensolver and set various options
    //Create eigensolver context
    EPS eps; /* eigenproblem solver context */
    EPSCreate(PETSC_COMM_WORLD,&eps);
    
    //Set operators. In this case, it is a standard eigenvalue problem */
    EPSSetOperators(eps,fA_Direct,NULL);
    EPSSetProblemType(eps,EPS_NHEP);
    EPSSetFromOptions(eps);
    EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);
    
    //Solve
    EPSSolve(eps);
    //EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);
    
    int nconv;
    double kr, ki; //real and imaginary part of the first eigenvalue
    EPSGetConverged( eps, &nconv ); //get the number of available eigenvalues
    for (int j=0; j<1; j++) {
        EPSGetEigenpair( eps, j, &kr, &ki, PETSC_NULL, PETSC_NULL );
    }
    
    EPSDestroy(&eps);

    double lambda=sqrt(kr*kr+ki*ki);
    
    return lambda;
}


double EigenAnalysisT::LargestEigen_Ave()
{
    FormA_Ave();
    EPS eps_1; /* eigenproblem solver context */
    EPSCreate(PETSC_COMM_WORLD,&eps_1);
    
    //Set operators. In this case, it is a standard eigenvalue problem */
    EPSSetOperators(eps_1,fA_Ave,NULL);
    EPSSetProblemType(eps_1,EPS_NHEP);
    EPSSetFromOptions(eps_1);
    EPSSetWhichEigenpairs(eps_1,EPS_LARGEST_MAGNITUDE);
    
    //Solve
    EPSSolve(eps_1);
    //EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);
    
    int nconv_1;
    double kr_1, ki_1; //real and imaginary part of the first eigenvalue
    EPSGetConverged( eps_1, &nconv_1 ); //get the number of available eigenvalues
    for (int j=0; j<1; j++) {
        EPSGetEigenpair( eps_1, j, &kr_1, &ki_1, PETSC_NULL, PETSC_NULL );
    }
    
    EPSDestroy(&eps_1);
    
    double lambda_1=sqrt(kr_1*kr_1+ki_1*ki_1);
    cout << "lambda_transpose="<<lambda_1<<endl;
    
    
    /*Mat A_shell;
    
    int NumARow=(fNumMatrix+1)*fNumRow;
    MatCreateShell(PETSC_COMM_WORLD, PETSC_DETERMINE,  PETSC_DETERMINE, NumARow, NumARow, this, &A_shell);
    MatSetFromOptions(A_shell);
    MatShellSetOperation(A_shell, MATOP_MULT, (void(*)())EigenAnalysisT::MatMult_A_Shell);
    
    EPS eps;
    EPSCreate(PETSC_COMM_WORLD,&eps);
    
    EPSSetOperators(eps,A_shell,NULL);
    EPSSetProblemType(eps,EPS_NHEP);
    EPSSetFromOptions(eps);
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
    
    EPSSolve(eps);
    
    EPSType        type;
    EPSGetType(eps,&type);
    PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);
    
    int nconv;
    double kr, ki; //real and imaginary part of the first eigenvalue
    EPSGetConverged( eps, &nconv ); //get the number of available eigenvalues
    for (int j=0; j<1; j++) {
        EPSGetEigenpair( eps, j, &kr, &ki, PETSC_NULL, PETSC_NULL );
    }
    
    double lambda=sqrt(kr*kr+ki*ki);
    cout << "lambda="<<lambda<<endl;
    
    EPSDestroy(&eps);
    MatDestroy(&A_shell);*/
    
    return lambda_1;
}


void EigenAnalysisT::InverseMPI(Mat A, Mat& A_Inv)
{
    int m, n;
     MatGetSize(A, &m, &n);
     
     if (m != n) {
     throw "!!!!! EignAnalysisT::InverseMPI, the matrix must be squre";
     }
    
    //collect a parallel matrix
    //step 1: get range
    int low, high;
    MatGetOwnershipRange(A, &low, &high);
    high=high-1;
    
    int size, rank;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    int* low_gather=new int[size];
    int* high_gather=new int[size];
    
    MPI_Gather(&low, 1, MPI_INT, low_gather, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Gather(&high, 1, MPI_INT, high_gather, 1, MPI_INT, 0, PETSC_COMM_WORLD);

    //step2: prepare send buffer
    int loc_num_row=high-low+1;
    int loc_count=loc_num_row*n;
    double* A_pointer_loc=new double[loc_count];
    
    int* row=new int[loc_num_row];
    int* column=new int[n];
    for (int i=low; i<=high; i++) {
        row[i-low]=i;
    }
    for (int i=0; i<n; i++) {
        column[i]=i;
    }
    
    MatGetValues(A, loc_num_row, row, n, column, A_pointer_loc);
    
    //step 3: prepare receive buffer
    double* A_gather_pointer=NULL;
    if (rank==0) {
        A_gather_pointer=new double[n*n];
    }
    
    int* counts=new int[size];
    int* disps=new int[size];
    
    for (int ri=0; ri<size; ri++) {
        counts[ri]=(high_gather[ri]-low_gather[ri]+1)*n;
    }
    disps[0]=0;
    for (int ri=1; ri<size; ri++) {
        disps[ri]=disps[ri-1]+counts[ri-1];
    }
    
    MPI_Gatherv(A_pointer_loc, loc_count, MPI_DOUBLE, A_gather_pointer, counts, disps, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    //step 4: do the inverse
    int* index=new int[m];
    for (int i=0; i<m; i++) {
        index[i]=i;
    }
    
    if (rank==0) {
        
        Mat A_seq;
        MatCreateSeqDense(PETSC_COMM_SELF,m,m,PETSC_NULL, &A_seq);
        MatSetValues(A_seq, m, index, m, index, A_gather_pointer, INSERT_VALUES);
        MatAssemblyBegin(A_seq, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A_seq, MAT_FINAL_ASSEMBLY);
        
        Mat A_inv_seq;
        MatCreateSeqDense(PETSC_COMM_SELF,m,m,PETSC_NULL, &A_inv_seq);
        
        Mat I_Matrix;
        MatCreateSeqDense(PETSC_COMM_SELF,m,m,PETSC_NULL, &I_Matrix);
        MatZeroEntries(I_Matrix);
        MatAssemblyBegin(I_Matrix, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(I_Matrix, MAT_FINAL_ASSEMBLY);
        
        MatShift(I_Matrix, 1.0);
        MatAssemblyBegin(I_Matrix, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(I_Matrix, MAT_FINAL_ASSEMBLY);
        
        MatLUFactor(A_seq,0,0,0);
        MatMatSolve(A_seq, I_Matrix, A_inv_seq);
        
        MatGetValues(A_inv_seq, m, index, m, index, A_gather_pointer);
        MatSetValues(A_Inv, m, index, m, index, A_gather_pointer, INSERT_VALUES);
        
        
        MatDestroy(&I_Matrix);
        MatDestroy(&A_seq);
        MatDestroy(&A_inv_seq);
    }
    
    MatAssemblyBegin(A_Inv, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A_Inv, MAT_FINAL_ASSEMBLY);
    
    
    delete [] low_gather;
    delete [] high_gather;
    delete [] A_pointer_loc;
    if (rank==0) delete [] A_gather_pointer;
    delete [] row;
    delete [] column;
    delete [] counts;
    delete [] disps;
    delete [] index;
    
    /*int m, n;
    MatGetSize(A, &m, &n);
    
    if (m != n) {
        throw "!!!!! EignAnalysisT::InverseMPI, the matrix must be squre";
    }
        
    //The following codes invert the matrix column by column
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    
    Vec I; //A column in indentity matrix
    Vec X;
    
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m, &I);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m, &X);
    
    int p,q;
    VecGetOwnershipRange(X, &p, &q);
    
    int* row_position=new int[q-p];
    for (int a=0; a<q-p; a++) {
        row_position[a]=p+a;
    }
    
    double* X_local;
    
    for (int i=0; i<m; i++) {
        VecZeroEntries(I);
        VecSetValue(I, i, 1, INSERT_VALUES);
        VecAssemblyBegin(I);
        VecAssemblyEnd(I);
        
        KSPSolve(ksp, I, X);
        
        VecGetArray(X, &X_local);
        
        MatSetValues(A_Inv, q-p, row_position, 1, &i, X_local, INSERT_VALUES);

        
        MatAssemblyBegin(A_Inv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A_Inv, MAT_FINAL_ASSEMBLY);
        
        VecRestoreArray(X, &X_local);
    }
    
    VecDestroy(&I);
    VecDestroy(&X);
    KSPDestroy(&ksp);
    
    delete [] row_position;*/
    
}


//This function is implemented to take the place of MatMatMult in Petsc.
//If MatMatMult can work for dense matrices, this function can be replaced.
void EigenAnalysisT::DenseMatMult(Mat A, Mat B, Mat& C)
{
    int m, n;
    MatGetSize(A, &m, &n);
    
    int p, q;
    MatGetSize(B, &p, &q);
    
    if (n != p) {
        throw "!!!!! EignAnalysisT::DenseMatMult, matrix dimension does not match";
    }
    
    Vec B_column; //One column of B
    Vec C_column; //One column of C
    
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, p, &B_column);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m, &C_column);
    
    int lower, upper;
    VecGetOwnershipRange(C_column, &lower, &upper);
    upper -= 1;
    int local_size=upper-lower+1;
    int* row_index=new int[local_size];
    
    for (int i=0; i<local_size; i++) {
        row_index[i]=lower+i;
    }
    
    //loop over columns
    for (int j=0; j<q; j++) {
        MatGetColumnVector(B, B_column, j);
        
        MatMult(A, B_column, C_column);
    
        //MatView(A, PETSC_VIEWER_STDOUT_WORLD);
        //VecView(B_column, PETSC_VIEWER_STDOUT_WORLD);
        //VecView(C_column, PETSC_VIEWER_STDOUT_WORLD);
        
        double* C_pointer;
        VecGetArray(C_column, &C_pointer);
        MatSetValues(C, local_size, row_index, 1, &j, C_pointer, INSERT_VALUES);
        VecRestoreArray(C_column, &C_pointer);
        
        MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);
        //MatView(C, PETSC_VIEWER_STDOUT_WORLD);
        
    }

    
    VecDestroy(&B_column);
    VecDestroy(&C_column);
    delete [] row_index;
    
}



void EigenAnalysisT::MatMult_A_Shell(Mat A, Vec x, Vec y)
{
    //Decompose Vector
    EigenAnalysisT* EA = NULL;
    MatShellGetContext(A, (void**)&EA);
    
    
    int NumMatrix=EA->GetNumMatrix();
    Mat* H_Ave=EA->Get_H_Ave();
    
    int m_local, n_local;
    MatGetSize(*H_Ave, &m_local, &n_local);
    
    Vec* sub_x=new Vec[NumMatrix+1];
    Vec* sub_y=new Vec[NumMatrix+1];
    for (int i=0; i<NumMatrix+1; i++) {
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m_local, sub_x+i);
        VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m_local, sub_y+i);
    }
    
    EA->DecomposeVector(x, NumMatrix+1, sub_x);
    
    //form the fisrt row
    VecZeroEntries(sub_y[0]);
    
    
    Vec y_temp;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m_local, &y_temp);
    for (int i=0; i<NumMatrix+1; i++) {
        
        VecCopy(sub_y[0], y_temp);
        MatMultAdd(H_Ave[i], sub_x[i], y_temp, sub_y[0]);
    }
    
    //form the remaining rows
    for (int i=1; i<NumMatrix+1; i++) {
        VecCopy(sub_x[i-1], sub_y[i]);
    }
    
    EA->CombineVector(NumMatrix+1, sub_y, y);
    
    
    for (int i=0; i<NumMatrix+1; i++) {
        VecDestroy(sub_x+i);
        VecDestroy(sub_x+i);
    }
    
    delete [] sub_x;
    delete [] sub_y;
    
    
    //VecView(y, PETSC_VIEWER_STDOUT_WORLD);
}

void EigenAnalysisT::DecomposeVector(Vec V, int m, Vec *sub_V)
{
    int length;
    VecGetSize(V, &length);
    
    int sub_length;
    VecGetSize(*sub_V, &sub_length);
    
    int mod=length%m;
    if (mod != 0) {
        throw "EigenAnalysisT::DecomposeVector, V cannot be divided into m sub vectors";
    }
    
    int low, high;
    VecGetOwnershipRange(V, &low, &high);
    high -=1;
    
    int local_length=high-low+1;
    
    const double* V_pointer;
    VecGetArrayRead(V, &V_pointer);
    
    for (int i=0; i<local_length; i++) {
        int global_id=low+i;
        int vec_id=floor(global_id/sub_length);
        int sub_id=global_id%sub_length;
        
        VecSetValue(sub_V[vec_id], sub_id, V_pointer[i], INSERT_VALUES);
    }
    
    
    for (int i=0; i<m; i++) {
        VecAssemblyBegin(sub_V[i]);
        VecAssemblyEnd(sub_V[i]);
    }
    
    VecRestoreArrayRead(V, &V_pointer);
    
    /*cout <<"Gross V is: "<<endl;
    VecView(V, PETSC_VIEWER_STDOUT_WORLD);
    
    for (int i=0; i<m; i++) {
        cout << i << "th sub V is: "<<endl;
        
        VecView(sub_V[i], PETSC_VIEWER_STDOUT_WORLD);
    }*/
    
}



void EigenAnalysisT::CombineVector(int m, Vec *sub_V, Vec V)
{
    int global_length;
    VecGetSize(V, &global_length);
    
    int sub_length;
    VecGetSize(*sub_V, &sub_length);
    
    if (m*sub_length != global_length) {
        throw "EigenAnalysisT::CombineVector, dimensions do not match!!";
    }
    

    int low, high;
    VecGetOwnershipRange(*sub_V, &low, &high);
    high -=1;
    
    int local_length=high-low+1;
    int* index=new int [local_length];
    
    for (int i=0; i<m; i++) {
        int head=i*sub_length+low;
        
        for (int j=0; j<local_length; j++) {
            index[j]=head+j;
        }
        
        const double* sub_V_pointer;
        VecGetArrayRead(sub_V[i], &sub_V_pointer);
        
        VecSetValues(V, local_length, index, sub_V_pointer, INSERT_VALUES);
        VecRestoreArrayRead(sub_V[i], &sub_V_pointer);
    }
    
    VecAssemblyBegin(V);
    VecAssemblyEnd(V);
    
    
    delete [] index;
    
    /*cout <<"Gross V is: "<<endl;
    VecView(V, PETSC_VIEWER_STDOUT_WORLD);
    
    for (int i=0; i<m; i++) {
        cout << i << "th sub V is: "<<endl;
        
        VecView(sub_V[i], PETSC_VIEWER_STDOUT_WORLD);
    }*/

}


int EigenAnalysisT::GetNumMatrix()
{
    return fNumMatrix;
}


Mat* EigenAnalysisT::Get_H_Ave()
{
    return fH_Ave;
}





