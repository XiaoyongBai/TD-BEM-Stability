// GHReaderT.cpp
//
//  Created by Xiaoyong Bai on 11/08/16.


#include "GHReaderT.h"

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

GHReaderT::GHReaderT()
{
    //initialize members
    fNumMatrix=0;
    fNumRow=0;
    
    fH1=NULL;
    fH2=NULL;
    fG1=NULL;
    fG2=NULL;
    
    fH=NULL;
    fG=NULL;
}


GHReaderT::~GHReaderT()
{
    if (fH1) delete[] fH1;
    if (fH2) delete[] fH2;
    if (fG1) delete[] fG1;
    if (fG2) delete[] fG2;
    
    if (fH) delete [] fH;
}


int GHReaderT::GetMatrixNumber()
{
    return fNumMatrix;
}

int GHReaderT::GetMatrixNumRow()
{
    return fNumRow;
}


void GHReaderT::ReadMatrixNumber()
{
    fNumMatrix=0;
    
    int step=0;
    int found=1;
    
    do{
        string G1_file, G2_file, H1_file, H2_file;
        
        ostringstream convert_G1, convert_G2, convert_H1, convert_H2; // stream used for the conversion
        
        convert_G1 << "G1_step" <<step<<".txt";
        G1_file=convert_G1.str();
        convert_G2 << "G2_step" <<step<<".txt";
        G2_file=convert_G2.str();
        convert_H1 << "H1_step" <<step<<".txt";
        H1_file=convert_H1.str();
        convert_H2 << "H2_step" <<step<<".txt";
        H2_file=convert_H2.str();
        
        
        ifstream input;
        
        input.open(G1_file.c_str(),ios_base::in);
        if (!input.good()) {
            found=0;
            break;
        }
        input.close();
        
        input.open(G2_file.c_str(),ios_base::in);
        if (!input.good()) {
            found=0;
            break;
        }
        input.close();
 
        input.open(H1_file.c_str(),ios_base::in);
        if (!input.good()) {
            found=0;
            break;
        }
        input.close();
        
        input.open(H2_file.c_str(),ios_base::in);
        if (!input.good()) {
            found=0;
            break;
        }
        input.close();
        
        fNumMatrix++;
        step++;
        
    }while (found==1);
}


void GHReaderT::ReadMatrixNumRow()
{
    string G1_file;
    ostringstream convert_G1;
    
    convert_G1 << "G1_step" <<1<<".txt";
    G1_file=convert_G1.str();
    
    ifstream input;
    
    input.open(G1_file.c_str(),ios_base::in);
    
    if (input.good()) {
        int num_row=0;
        int num_colum_old=0;
        int num_colum_new=0;
        
        const int MAX_CHARS_LINE=100000;
        const char* const DELIMITER = " \t";
        
        char bf[MAX_CHARS_LINE];
        char* token[1]={};
        
        while (!input.eof()) {
            
            input.getline(bf, MAX_CHARS_LINE);
            token[0]=strtok(bf, DELIMITER);
            
            if (token[0]) {
                num_row++;
                
                if (num_colum_old!=0 && num_colum_old!=num_colum_new)
                    cout<< "Error: matrix column is inconsitent \n";
            }
            
            num_colum_old=num_colum_new;
            num_colum_new=0;
            
            while(token[0]){
                num_colum_new++;
                
                token[0]=strtok(NULL, DELIMITER);
            }
                   
        }
        
        if (num_colum_old!=num_row)
            cout<<"Error: number of row and column is not identical \n";
        
        
        fNumRow=num_row;

    }
    input.close();

}

void GHReaderT::ReadMatrices()
{
    Mat AA;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &AA);
    
    int lower, upper;
    MatGetOwnershipRange(AA, &lower, &upper);
    upper -= 1;
    MatDestroy(&AA);
    
    //Allocate matrices
    int local_num_row= upper-lower+1;
    int single_size= local_num_row*fNumRow; //size of one matrix
    int total_size= fNumMatrix*single_size;
    
    //int rank;
    //MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    //cout << "rank "<<rank <<" lower="<<lower<< " upper="<<upper<<endl;
    //cout << "rank "<<rank <<" local_num_row="<<local_num_row<<endl;
    //cout << "rank "<<rank <<" single_size="<<single_size<<endl;
    //cout << "rank "<<rank <<" total_size="<<total_size<<endl;
    
    fH1=new double[total_size];
    fH2=new double[total_size];
    fG1=new double[total_size];
    fG2=new double[total_size];
    
    //read fH1
    for (int mi=0; mi<fNumMatrix; mi++) {
        string H1_file;
        ostringstream convert_H1;
        
        convert_H1 << "H1_step" << mi <<".txt";
        H1_file=convert_H1.str();
        
        ReadMatrix(fH1+mi*single_size, lower, upper, H1_file.c_str());
        
        /*string out_file;
        ostringstream convert_out;
        
        convert_out << "H1_step_" << mi << "_rank_"<<rank<< "_.txt";
        out_file=convert_out.str();
        
        ofstream write;
        
        write.open(out_file.c_str(),ios_base::out);
    
        for (int i=0; i<local_num_row; i++) {
            for (int j=0; j<fNumRow; j++) {
                write<< setw(15) << setprecision(5) << std::scientific << fH1[mi*single_size+i*fNumRow+j];
            }
            write<<endl;
        }
        
        write.close();*/
    }
    
    //read fH2
    for (int mi=0; mi<fNumMatrix; mi++) {
        string H2_file;
        ostringstream convert_H2;
        
        convert_H2 << "H2_step" << mi <<".txt";
        H2_file=convert_H2.str();
        
        ReadMatrix(fH2+mi*single_size, lower, upper, H2_file.c_str());
    }
    
    //read fG1
    for (int mi=0; mi<fNumMatrix; mi++) {
        string G1_file;
        ostringstream convert_G1;
        
        convert_G1 << "G1_step" << mi <<".txt";
        G1_file=convert_G1.str();
        
        ReadMatrix(fG1+mi*single_size, lower, upper, G1_file.c_str());
    }
    
    //read fG2
    for (int mi=0; mi<fNumMatrix; mi++) {
        string G2_file;
        ostringstream convert_G2;
        
        convert_G2 << "G2_step" << mi <<".txt";
        G2_file=convert_G2.str();
        
        ReadMatrix(fG2+mi*single_size, lower, upper, G2_file.c_str());
    }
    
    //group H
    int H_size=(fNumMatrix+1)*single_size;
    fH=new double[H_size];
    
    for (int i=0; i<single_size; i++) {
        fH[i]=fH1[i];
    }
    
    for (int i=1; i<fNumMatrix; i++) {
        
        int h2_start=(i-1)*single_size;
        int h2_end=i*single_size;
        for (int j=h2_start; j<h2_end; j++) {
            fH[j+single_size]=fH2[j];
        }
        
        int h1_start=i*single_size;
        int h1_end=(i+1)*single_size;
        for (int j=h1_start; j<h1_end; j++) {
            fH[j]+=fH1[j];
        }
    }
    
    int h2_start=(fNumMatrix-1)*single_size;
    int h2_end=fNumMatrix*single_size;
    for (int j=h2_start; j<h2_end; j++) {
        fH[j+single_size]=fH2[j];
    }
    
    
    //group G
    int G_size=(fNumMatrix+1)*single_size;
    fG=new double[G_size];
    
    for (int i=0; i<single_size; i++) {
        fG[i]=fG1[i];
    }
    
    for (int i=1; i<fNumMatrix; i++) {
        
        int g2_start=(i-1)*single_size;
        int g2_end=i*single_size;
        for (int j=g2_start; j<g2_end; j++) {
            fG[j+single_size]=fG2[j];
        }
        
        int g1_start=i*single_size;
        int g1_end=(i+1)*single_size;
        for (int j=g1_start; j<g1_end; j++) {
            fG[j]+=fG1[j];
        }
    }
    
    int g2_start=(fNumMatrix-1)*single_size;
    int g2_end=fNumMatrix*single_size;
    for (int j=g2_start; j<g2_end; j++) {
        fG[j+single_size]=fG2[j];
    }
    
    delete [] fG1; fG1=NULL;
    delete [] fG2; fG2=NULL;
    delete [] fH1; fH1=NULL;
    delete [] fH2; fH2=NULL;
    
}


void GHReaderT::GetH(double **H)
{
    *H=fH;
}


void GHReaderT::ReadMatrix(double* H, int firstRow, int lastRow, const char* file)
{
    ifstream input;
    
    input.open(file,ios_base::in);
    
    if (!input.good()) {
        cout<< "GHReaderT::ReadMatrix() " << file <<endl;
        throw "cannot find file";
    }
    
    double a;
    
    //read useless part;
    for (int i=0; i<firstRow; i++) {
        for (int j=0; j<fNumRow; j++) {
            input>> a;
        }
    }
    
    //read the effective part;
    for (int i=firstRow; i<lastRow+1; i++) {
        for (int j=0; j<fNumRow; j++) {
            input>> H[(i-firstRow)*fNumRow+j];
        }
    }
    
    input.close();
}



void GHReaderT::GetMatrices(double** H1, double** H2, double** G1, double** G2)
{
    *H1=fH1;
    *H2=fH2;
    *G1=fG1;
    *G2=fG2;
}

void GHReaderT::ExchangeColumn_GH()
{
    ifstream input;
    
    input.open("UBC.txt",ios_base::in);
    
    if (!input.good()) {
        cout<< "GHReaderT::ExchangeColumn_GH() " << "UBC.txt" <<endl;
        throw "cannot find file";
    }
    
    int ubc_num=0;
    input >> ubc_num;
    
    int* ubc=new int [ubc_num];
    
    for (int i=0; i<ubc_num; i++) {
        input>>ubc[i];
        ubc[i] -= 1;
    }
    
    input.close();
    
    Mat AA;
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, fNumRow, fNumRow, PETSC_NULL, &AA);
    
    int lower, upper;
    MatGetOwnershipRange(AA, &lower, &upper);
    upper -= 1;
    MatDestroy(&AA);
    
    //Compute matrix dimesions
    int local_num_row= upper-lower+1;
    int totoal_num_row= local_num_row*(fNumMatrix+1); //number of rows of the global matrix
    
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    
    /*if (rank==0) {
        cout <<"\n"<<"Matrix H is"<<endl;
        
        for (int i=0; i<totoal_num_row; i++) {
            for (int j=0; j<fNumRow; j++) {
                cout <<setw(5) << fH[i*fNumRow+j] << " ";
            }
            cout<<endl;
        }
        
        cout <<"\n"<<"Matrix G is"<<endl;
        for (int i=0; i<totoal_num_row; i++) {
            for (int j=0; j<fNumRow; j++) {
                cout <<setw(5) << fG[i*fNumRow+j] << " ";
            }
            cout<<endl;
        }
    }*/

    for (int i=0; i<ubc_num; i++) {
        for (int j=0; j<totoal_num_row; j++) {
            int dof=ubc[i];
            fH[j*fNumRow+i] = -fG[j*fNumRow+dof];
        }
    }
    
    
    if (rank==0) {
        cout <<"\n"<<"Matrix H is"<<endl;
        
        for (int i=0; i<totoal_num_row; i++) {
            for (int j=0; j<fNumRow; j++) {
                cout <<setw(5) << fH[i*fNumRow+j] << " ";
            }
            cout<<endl;
        }
    }
    
    
    delete [] ubc; ubc=NULL;
    delete [] fG;  fG=NULL;
}

