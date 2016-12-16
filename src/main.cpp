
#include "slepcsys.h"
#include <slepceps.h>
#include "petscmat.h" 

#include "iostream"
#include "iomanip"
#include "fstream"
#include "GHReaderT.h"
#include "EigenAnalysisT.h"

using namespace std;
using namespace Stability;

void TestInvsere();

int main( int argc, char **argv )
{
    char help[] = "Eigenvalue analysis of BEM matrix, usingSLEPc\n";
    SlepcInitialize(&argc,&argv,(char*)0,help);
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank );
    
    PetscErrorCode ierr;
    try {
        
        //TestInvsere();
        //return 0;
        
        //read matrices
        GHReaderT reader;
        reader.ReadMatrixNumber();
        reader.ReadMatrixNumRow();
        
        int num_matrix=reader.GetMatrixNumber();
        int num_row=reader.GetMatrixNumRow();
        
        cout<<"num matrix="<<num_matrix<<endl;
        cout<<"num row="<<num_row<<endl;
    
        if (rank==0) {
            reader.ReadMatrices();
        }
        MPI_Barrier(PETSC_COMM_WORLD);
        
        double* H;
        reader.GetH(&H);
        
        EigenAnalysisT EA;
        EA.SetMatrixNumSize(num_matrix, num_row);
        
        EA.SetMatrixSystem_Direct(H);
        
        double aa = EA.LargestEigen_Direct();
        cout << "\n the largest eigenvalue of A for direct stepping=" << aa <<endl;

        double A1=0.25;
        double A2=0.25;
        EA.SetMatrixSystem_Ave(H, A1, A2);
        
        double bb=EA.LargestEigen_Ave();
        
        cout << "\n the largest eigenvalue of A for averaging stepping=" << bb <<endl;
        

        
        ofstream write_eig;
        
        if ( rank==0 ) {
            write_eig.open("eig.txt",ios_base::out);
            write_eig <<setw(15)<<"% alpha1"<<setw(15)<<"alpha2"<<setw(15)<<"Eigenvalue"<<endl;
        }
    
         for (double a1=0.0; a1<=1.0; a1+=0.1) {
             for (double a2=0.0; a2<=1.0-a1; a2+=0.1) {
         
                 if (a2==1.0) continue;
                 
                 EA.SetMatrixSystem_Ave(H, a1, a2);
         
                 double eig=EA.LargestEigen_Ave();
                 
                 if (rank==0) {
                     write_eig<<setw(15) <<setprecision(5) << std::scientific << a1;
                     write_eig<<setw(15) <<setprecision(5) << std::scientific << a2;
                     write_eig<<setw(15) <<setprecision(5) << std::scientific << eig;
                     write_eig<<endl;
                 }
                 
             }
         }
         
        if (rank==0) write_eig.close();
        
        
    } catch (const char* msg) {
        cerr<<msg<<endl;
        return 0;
    }
    
    
    
    
    ierr = SlepcFinalize();
    return 1;
}




void TestInvsere()
{
    //test outputASCII
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank );
    
    cout <<"rank is" << rank <<endl;
    
    Mat AA_global;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 5, 5, PETSC_NULL, &AA_global);
    
    if (rank==0) {
        Mat AA;
        PetscViewer fd;
        MatCreateSeqDense(PETSC_COMM_SELF, 5, 5, PETSC_NULL, &AA);
        
        MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY);
        MatShift(AA, 2.0);
        MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY);
        
        //MatView(AA, PETSC_VIEWER_STDOUT_SELF);

        //Generate an Identity matrix.
        Mat I_Matrix;
        MatCreateSeqDense(PETSC_COMM_SELF,5,5,PETSC_NULL, &I_Matrix);
        MatZeroEntries(I_Matrix);
        MatAssemblyBegin(I_Matrix, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(I_Matrix, MAT_FINAL_ASSEMBLY);
        MatShift(I_Matrix, -1.0);
        MatAssemblyBegin(I_Matrix, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(I_Matrix, MAT_FINAL_ASSEMBLY);
        
        //MatView(I_Matrix, PETSC_VIEWER_STDOUT_SELF);
        
        //Compute inverse of AA.
        Mat AA_Inv;
        MatCreateSeqDense(PETSC_COMM_SELF, 5, 5, PETSC_NULL, &AA_Inv);
        MatAssemblyBegin(AA_Inv, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(AA_Inv, MAT_FINAL_ASSEMBLY);
        
        MatLUFactor(AA,0,0,0);
        MatMatSolve(AA, I_Matrix, AA_Inv);
        
        MatView(AA_Inv, PETSC_VIEWER_STDOUT_SELF);
        
        
        double* vec=new double[25];
        int* column=new int[5];
        for (int i=0; i<5; i++) {
            column[i]=i;
        }
        MatGetValues(AA_Inv,5,column,5,column, vec);
        
        MatSetValues(AA_global, 5, column, 5, column, vec, INSERT_VALUES);
        
    }
    
    MatAssemblyBegin(AA_global, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA_global, MAT_FINAL_ASSEMBLY);
    
    cout <<">>>>>>>>>>>"<<endl;
    MatView(AA_global, PETSC_VIEWER_STDOUT_WORLD);

    Mat B;
    
    MatMatMult(AA_global, AA_global, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B);
    cout <<"<<<<<<<<<<<<<"<<endl;

     MatView(B, PETSC_VIEWER_STDOUT_WORLD);

    MPI_Barrier(PETSC_COMM_WORLD);
    
    /*MatZeroEntries(AA);
    MatAssemblyBegin(AA, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(AA, MAT_FINAL_ASSEMBLY);
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,"matrix.txt",&fd);
    MatView(AA, fd);
    PetscViewerDestroy(&fd);
    MatDestroy(&AA);*/
    


    
}
