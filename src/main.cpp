
#include "slepcsys.h"
#include <slepceps.h>
#include "petscmat.h" 

#include "iostream"
#include "iomanip"
#include "fstream"
#include "GHReaderT.h"
#include "EigenAnalysisT.h"

#include <ctime>

using namespace std;
using namespace Stability;

int main( int argc, char **argv )
{
    char help[] = "Eigenvalue analysis of BEM matrix, usingSLEPc\n";
    SlepcInitialize(&argc,&argv,(char*)0,help);
        
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank );
    
    PetscErrorCode ierr;
    try {
               
        //read matrices
        GHReaderT reader;
        reader.ReadMatrixNumber();
        reader.ReadMatrixNumRow();
        
        int num_matrix=reader.GetMatrixNumber();
        int num_row=reader.GetMatrixNumRow();
        
        cout<<"num matrix="<<num_matrix<<endl;
        cout<<"num row="<<num_row<<endl;
    
        reader.ReadMatrices();
        reader.ExchangeColumn_GH();
        
        MPI_Barrier(PETSC_COMM_WORLD);
        
        double* H;
        reader.GetH(&H);
        
        EigenAnalysisT EA;
        EA.SetMatrixNumSize(num_matrix, num_row);
        
        std::time_t result;
        double A1=0.25;
        double A2=0.25;
        
        result = std::time(NULL);
        std::cout << std::asctime(std::localtime(&result)) <<endl;
        
        EA.SetMatrixSystem_Ave(H, A1, A2);
        
        result = std::time(NULL);
        std::cout << std::asctime(std::localtime(&result)) <<endl;
        
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
                 
                 result = std::time(NULL);
                 std::cout << std::asctime(std::localtime(&result)) <<endl;
                 
                 EA.SetMatrixSystem_Ave(H, a1, a2);
         
                 result = std::time(NULL);
                 std::cout << std::asctime(std::localtime(&result)) <<endl;
                 
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
