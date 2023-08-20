#include<bits/stdc++.h>
using namespace std;
string rec(int x, map<int,vector<int>> decrypt){
	if(decrypt.find(x)!=decrypt.end()){
		string ans = "";
		for(auto item: decrypt[x]){
			ans += rec(item,decrypt) + " ";
		}
		return ans;
	}
	return to_string(x);
}
void getback_file(string inputfile, string outputfile){
	
	//Obtaining map
	fstream inp(inputfile);
	string s;
	map<int,vector<int>> decrypt;
	while(getline(inp,s)){
		if(s.empty()){
			break;
		}
		continue;
	}
	while(getline(inp,s)){
		stringstream ss(s);
		string temp;
		int curkey;
		ss>>temp;
		curkey = stoi(temp);
		while(ss>>temp){
			decrypt[curkey].push_back(stoi(temp));
		}
	}
	inp.close();

	// using map to create original file
	ofstream output;
	output.open(outputfile);
	fstream inp1(inputfile);
	s.clear();
	while(getline(inp1,s)){
		if(s.empty()){
			break;
		}
		stringstream ss1(s);
		string temp;
		while(ss1>>temp){
			if(decrypt.find(stoi(temp))!=decrypt.end()){
				string val = rec(stoi(temp),decrypt);
				output<<val;
			}
			else{
				output<< temp << " ";
			}
		}
		output<< "\n";
	}
	inp1.close();
	output.close();


}
int main(int argc, char* argv[]){
    string infile = argv[1];
    string outfile = argv[2];
    getback_file(infile,outfile);
    
    return 0;
}