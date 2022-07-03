#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
using namespace std;

const int BATCH_SIZE = 20;

struct xorshift{
	unsigned x, y, z, w;
	xorshift(): x(1234578), y(34578920), z(23409889), w(12309837){}
	
	unsigned next(){
		unsigned t = x^(x<<11);
		
		x=y; y=z; z=w;
		
		return w = w^(w>>19)^t^(t>>8);
	}
} random;

double random_01(){
	return random.next()%1000000001 / 1000000000.0;
}

class matrix{
	public:
		int n, m;
		double **a;
		//default constructor
		matrix(): n(0), m(0), a(NULL){}
		
		//constructor
		matrix(int rows, int cols){
			n = rows;
			m = cols;
			a = new double*[n];
			
			for(int i=0;i<n;i++){
				a[i] = new double [m];
				for(int j=0;j<m;j++){
					a[i][j] = 0.0;
				}
			}
		}
		
		//copy constructor
		matrix(const matrix &x){
			n = x.n;
			m = x.m;
			
			a = new double*[n];
			for(int i=0;i<n;i++){
				a[i] = new double[m];
				for(int j=0;j<m;j++){
					a[i][j] = x.a[i][j];
				}
			}
		}
		//destructor
		~matrix(){
			for(int i=0;i<n;i++){
				delete [] a[i];
			}
			delete [] a;
		}
		//random initialization
		void randomize(){
			for(int i=0;i<n;i++){
				for(int j=0;j<m;j++){
					a[i][j] = random_01();
					if(random.next()%2 == 0) a[i][j] = -a[i][j];
				}
			}
		}
		//matrix of zeros
		void zero(){
			for(int i=0;i<n;i++){
				for(int j=0;j<m;j++){
					a[i][j] = 0.0;
				}
			}	
		}
		
		void add(const matrix &x){
			for(int i=0;i<n;i++){
				for(int j=0;j<m;j++){
					a[i][j] += x.a[i][j];
				}		
			}
		}
		
		//= operator
		void operator = (const matrix &x){
			int n = x.n;
			int m = x.m;
			
			a = new double*[n];
			for(int i=0;i<n;i++){
				a[i] = new double[m];
				for(int j=0;j<m;j++){
					a[i][j] = x.a[i][j];
				}
			}
		}
		
		//matrix[i][j]
		double* operator [] (const int &idx){
			return a[idx];
		}
};

//assuming both matrix has the same dimensions
matrix add(matrix A, matrix B){
	matrix result(A.n, A.m);
	for(int i=0;i<A.n;i++){
		for(int j=0;j<A.m;j++){
			result[i][j] = A[i][j] + B[i][j];
		}
	}
	return result;
}

matrix subtract(matrix A, matrix B){
	matrix result(A.n, A.m);
	for(int i=0;i<A.n;i++){
		for(int j=0;j<A.m;j++){
			result[i][j] = A[i][j] - B[i][j];
		}
	}
	return result;
}

matrix element_wise_mult(matrix A, matrix B){
	matrix result(A.n, A.m);
	for(int i=0;i<A.n;i++){
		for(int j=0;j<A.m;j++){
			result[i][j] = A[i][j] * B[i][j];
		}
	}
	return result;
}

matrix transpose(matrix A){
	matrix result(A.m, A.n);
	for(int i=0;i<A.n;i++){
		for(int j=0;j<A.m;j++){
			result[j][i] = A[i][j];
		}
	}
	return result;	
}

matrix multiply(matrix A, matrix B){
	matrix result(A.m, A.n);
	for(int i=0;i<A.n;i++){
		for(int j=0;j<A.m;j++){
			for(int k=0;k<A.m;k++){
				result[i][j] = A[i][k] + B[k][j];
			}
		}
	}
	return result;	
}

vector <int> split(string s){
	int i, curr=0, check=0;
	
	vector <int> ans;
	for(i = 0;i<(int)(s.size());i++){
		if(s[i]==','){
			ans.push_back(curr);
			curr = 0;
		}
		else{
			curr *= 10;
			curr += s[i] - '0';
		}
	}
	ans.push_back(curr);
	return ans;
}

void time_stamp(){
	cout<<"Time: "<<(int)(clock()*1000.0/CLOCKS_PER_SEC)<<" ms."<<endl;	
}

vector <matrix> train_input, train_output;
void parse_training_data(){
	ifstream IN("train.csv");
	string trash;
	vector <int> v;
	matrix input(1, 784), output(1, 10);
	
	train_input.reserve(42000);
	train_output.reserve(42000);
	
	IN>>trash;
	for(int i=0; i<42000;i++){
		IN>>trash;
		
		v = split(trash);
		//one-hot encoding
		output.zero();
		output[0][v[0]] = 1.0;
		
		for(int j=1; j<785; j++){
			input[0][j-1] = v[j] / 255.0;
		}
		
		train_input.push_back(input);
		train_output.push_back(output);
	}
	
	cout<<"Training Data load complete."<<endl;
	time_stamp();
}

int main(){
	parse_training_data();
}
