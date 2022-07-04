#include <iostream>
#include <conio.h>
#include <math.h>
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
	matrix result(A.n, B.m);
	//cout<<"A: "<<A.n<<","<<A.m<<endl;
	//cout<<"B: "<<B.n<<","<<B.m<<endl;
	for(int i=0;i<A.n;i++){
		for(int j=0;j<B.m;j++){
			for(int k=0;k<A.m;k++){
				result[i][j] = A[i][k] + B[k][j];
			}
		}
	}

	return result;	
}

double sigmoid_single(double x){
	return 1.0 / (1.0+exp(-x));
}

double sigmoid_derivative(double x){
	return x*(1.0-x);
}

matrix sigmoid(matrix a){
	matrix result(a.n, a.m);
	for(int i=0;i<a.n;i++){
		for(int j=0;j<a.m;j++){
			result[i][j] = sigmoid_single(a[i][j]);
		}
	}
	return result;		
}

matrix sigmoid_derivative(matrix a){
	matrix result(a.n, a.m);
	for(int i=0;i<a.n;i++){
		for(int j=0;j<a.m;j++){
			result[i][j] = sigmoid_derivative(a[i][j]);
		}
	}
	return result;	
}

class neural_network{
	public:
		int n;
		vector <int> size;
		vector <matrix> w, b, delta_w, delta_b;
		double learning_rate;
		
		neural_network(){}
		
		neural_network(vector <int> sz, double alpha){
			int i;
			n= (int)(sz.size());
			size = sz;
			
			w.resize(n-1);
			b.resize(n-1);
			delta_w.resize(n-1);
			delta_b.resize(n-1);
			
			
			for(int i=0; i<n-1; i++){
				w[i] = matrix(size[i], size[i+1]);
				b[i] = matrix(1, size[i+1]);
				delta_w[i] = matrix(size[i], size[i+1]);
				delta_b[i] = matrix(1, size[i+1]);
				
				w[i].randomize();
				b[i].randomize();
			}
			
			learning_rate = alpha;
		}
		
		matrix forward_prop(matrix &input){
			
			for(int i=0; i<n-1; i++){
				input = sigmoid(add(multiply(input, w[i]), b[i]));
			}
			return input;
		}
		
		void backpropagation(matrix input, matrix output){
			vector<matrix> l;
			matrix delta;
			int i;
			
			l.push_back(input);
			for(int i=0; i<n-1;i++){
				input = sigmoid(add(multiply(input, w[i]), b[i]));
				l.push_back(input);
			}
			
			delta = element_wise_mult(subtract(input, output), sigmoid_derivative(l[n-1]));
			delta_b[n-2].add(delta);
			delta_w[n-2].add(multiply(transpose(l[n-2]), delta));

			for(int i=n-3; i>=0; i--){
				delta = multiply(delta, transpose(w[i+1]));
				delta = element_wise_mult(delta, sigmoid_derivative(l[i+1]));
				
				delta_b[i].add(delta);
				delta_w[i].add(multiply(transpose(l[i]), delta));
			}
		}
		
		void train(vector <matrix> inputs, vector <matrix> outputs){
			for(int i=0; i<=n-1; i++){
				delta_w[i].zero();
				delta_b[i].zero();
			}
			
			for(int i=0; i<(int)(inputs.size()); i++){
				backpropagation(inputs[i], outputs[i]);
			}

			for(int i=0;i<n-1;i++){
				for(int j=0; j<delta_w[i].n;j++){
					for(int z=0; z<delta_w[i].m;z++){
						delta_w[i][j][z] /= (double)(inputs.size());
						w[i][j][z] -= learning_rate*delta_w[i][j][z];
					}
				}
			


				for(int j=0; j<delta_b[i].n;j++){
					for(int z=0; z<delta_b[i].m;z++){
						delta_b[i][j][z] /= (double)(inputs.size());
						b[i][j][z] -= learning_rate*delta_b[i][j][z];
					}
				}
			}
		}
};

vector <matrix> train_input, train_output;


vector <int> split(string s){
	int i, curr=0;
	
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

void random_shuffle(vector <int> &v){
	for(int i=(int)(v.size());i>=0;i--){
		swap(v[i], v[random.next()%(i+1)]);
	}	
}

void train(){
	vector <int> units;
	units.push_back(784);
	//units.push_back(15);
	units.push_back(10);
	neural_network net(units, 1.0);
	
	int epoch;
	vector <int> idx;
	vector <matrix> inputs, outputs;
	matrix curr_output;
	double error;
	
	for(int i=0; i<42000;i++){
		idx.push_back(i);
	}
	
	for(epoch=1;epoch<=10;epoch++){
		cout<<"Epoch: "<<epoch<<endl;
		error = 0.0;
		random_shuffle(idx);
		for(int i=0;i<42000;i+=BATCH_SIZE){
			inputs.clear();
			outputs.clear();
			
			for(int j=0;j<BATCH_SIZE;j++){
				inputs.push_back(train_input[idx[i+j]]);
				outputs.push_back(train_output[idx[i+j]]);
			}
			net.train(inputs, outputs);
		}
		
		for(int i=0;i<42000;i++){
			curr_output = net.forward_prop(train_input[i]);
			
			for(int j=0;j<10;j++){
				error += (curr_output[0][j]-train_output[i][0][j])*(curr_output[0][j]-train_output[i][0][j]);
			}
		}
		
		error /= 10.0;
		error /= 42000.0;
		
		cout<<"Epoch: "<<epoch<<" finished."<<endl;
		cout<<"Error: "<<error<<endl;
		time_stamp();
		cout<<endl;
	}
	//save weight
	ofstream OUT("weights.txt");
	OUT<<"w[0]"<<endl;
	for(int i=0; i<net.w[0].n; i++){
		for(int j=0; j<net.w[0].m;j++){
			OUT<<net.w[0][i][j]<<",";
		}
	}
	cout<<"Done w[0]"<<endl;
	OUT<<"\n";
	
	OUT<<"w[1]"<<endl;
	for(int i=0; i<net.w[1].n; i++){
		for(int j=0; j<net.w[1].m;j++){
			OUT<<net.w[1][i][j]<<",";
		}
	}
	cout<<"Done w[1]"<<endl;
	OUT<<"\n";
	
	OUT<<"b[0]"<<endl;
	for(int i=0; i<net.b[0].n; i++){
		for(int j=0; j<net.b[0].m;j++){
			OUT<<net.b[0][i][j]<<",";
		}
	}
	cout<<"Done b[0]"<<endl;
	OUT<<"\n";
	
	OUT<<"b[1]"<<endl;
	for(int i=0; i<net.b[1].n; i++){
		for(int j=0; j<net.b[1].m;j++){
			OUT<<net.b[1][i][j]<<",";
		}
	}
	cout<<"Done b[1]"<<endl;
	OUT<<"\n";
	

	OUT.close();
}


	
int main(){
	parse_training_data();
	train();
}
