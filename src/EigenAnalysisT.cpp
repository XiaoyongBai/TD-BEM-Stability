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
    
    
    //test inverse
    /*Mat AA;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 3, PETSC_NULL, &AA);
    MatZeroEntries(AA);
    MatSetValue(AA, 0, 1, 1, INSERT_VALUES);
    MatSetValue(AA, 1, 0, 3, INSERT_VALUES);
    MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY);
    MatShift(AA, 2.0);
    MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY);
    
    cout <<">>>>>>>>>>>"<<endl;
    MatView(AA, PETSC_VIEWER_STDOUT_WORLD);
    
    Mat BB;
    MatConvert(AA, MATSAME, MAT_INITIAL_MATRIX, &BB);
    InverseMPI(AA, BB);

    
    cout <<">>>>>>>>>>>"<<endl;
    MatView(BB, PETSC_VIEWER_STDOUT_WORLD);
    
    MatDestroy(&AA);
    MatDestroy(&BB);*/
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
        
        //MatMatMult(H0_Inv, H_temp, MAT_REUSE_MATRIX, PETSC_DEFAULT, &H_temp_1);
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
    
    cout <<">>>>>>>>>>>"<<endl;
    cout <<"H0 is"<<endl;
    MatView(H0, PETSC_VIEWER_STDOUT_WORLD);
    
    cout <<">>>>>>>>>>>"<<endl;
    cout <<"Inverse of H0 is"<<endl;
    MatView(H0_Inv, PETSC_VIEWER_STDOUT_WORLD);
        
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
    
    cout <<"fA_Direct is"<<endl;
    MatView(fA_Direct,PETSC_VIEWER_STDOUT_WORLD);
}


void EigenAnalysisT::FormA_Ave()
{
    int NumARow=(fNumMatrix+1)*fNumRow;
    
    //Allocate A
    if (!fA_Ave) {
        MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NumARow, NumARow, PETSC_NULL, &fA_Ave);
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
    int* column_global=new int[numColumn];
    for (int i=fHFirst; i<=fHLast; i++) {
        row[i-fHFirst]=i;
    }
    for (int i=0; i<numColumn; i++) {
        column[i]=i;
    }
    
    for (int hi=0; hi<fNumMatrix+1; hi++) {
        MatGetValues(fH_Ave[hi], numRow, row, numColumn, column, vec);
        
        for (int ci=0; ci<numColumn; ci++) {
            column_global[ci]=ci+hi*numColumn;
        }
        
        MatSetValues(fA_Ave, numRow, row, numColumn, column_global, vec, INSERT_VALUES);
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
                MatSetValue(fA_Ave, ri, ri-fNumRow, 1.0, INSERT_VALUES);
            }
        }
    }else{
        for (int ri=fAFirst_ave; ri<=fALast_ave; ri++) {
            MatSetValue(fA_Ave, ri, ri-fNumRow, 1.0, INSERT_VALUES);
        }
    }
    
    MatAssemblyBegin(fA_Ave, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(fA_Ave, MAT_FINAL_ASSEMBLY);
    
    //MatView(fA_Ave,PETSC_VIEWER_STDOUT_WORLD);
    
    delete [] column;
    delete [] row;
    delete [] column_global;
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
    
    //Create the eigensolver and set various options
    //Create eigensolver context
    EPS eps; /* eigenproblem solver context */
    EPSCreate(PETSC_COMM_WORLD,&eps);
    
    //Set operators. In this case, it is a standard eigenvalue problem */
    EPSSetOperators(eps,fA_Ave,NULL);
    EPSSetProblemType(eps,EPS_NHEP);
    EPSSetFromOptions(eps);
    EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);
    
    //Solve
    EPSSolve(eps);
    //EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);
    
    int nconv;
    double kr, ki, lambda; //real and imaginary part of the first eigenvalue
    EPSGetConverged( eps, &nconv ); //get the number of available eigenvalues
    
    if (nconv>=1) {
        for (int j=0; j<1; j++) {
            EPSGetEigenpair( eps, j, &kr, &ki, PETSC_NULL, PETSC_NULL );
        }
        lambda=sqrt(kr*kr+ki*ki);
    }else
    {
        lambda=-1.0;
    }

    
    EPSDestroy(&eps);
    
    return lambda;
}


void EigenAnalysisT::InverseMPI(Mat A, Mat& A_Inv)
{
    int m, n;
    MatGetSize(A, &m, &n);
    
    if (m != n) {
        throw "!!!!! EignAnalysisT::InverseMPI, the matrix must be squre";
    }
        
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
    
    delete [] row_position;
    
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

















