#include <bits/stdc++.h>
#include <string>
#define N 100050
using namespace std;

const double eps = 1e-5;

int num_of_walks = 40;
int seq_len = 4;
int start = 0;
int o[N];
char dis[N][N];
bool vis[N];
int n, m;

vector<int> E[N];
vector<double> p_merw[N];

class AliasTable{
    public:
        vector<int> A, B;
        vector<double> S;
    public:
        AliasTable () {}
        void init(vector<int> &a, vector<double> &p) {
            queue<int> qA, qB;
            queue<double> pA, pB;
            int n = (int)a.size();
            for (int i=0;i<n;i++) p[i] = p[i] * n;//*n?
            for (int i=0;i<n;i++)
                if (p[i] > 1.0) {
                    qA.push(a[i]);
                    pA.push(p[i]);  
                } else {
                    qB.push(a[i]);
                    pB.push(p[i]);
                }
            while (!qA.empty() && !qB.empty()) {
                int idA = qA.front(); 
                qA.pop();
                double probA = pA.front(); pA.pop();
                int idB = qB.front(); qB.pop();
                double probB = pB.front(); pB.pop();
                
                A.push_back(idA);
                B.push_back(idB);
                S.push_back(probB);

                double res = probA-(1.0-probB);

                if (abs(res-1.0) < eps) {
                    A.push_back(idA);
                    B.push_back(idA);
                    S.push_back(res);
                    continue;
                }

                if (res > 1.0) {
                    qA.push(idA);
                    pA.push(res);
                } else {
                    qB.push(idA);
                    pB.push(res);
                }
            }

            while (!qA.empty()) {
                int idA = qA.front(); qA.pop();
                pA.pop();
                A.push_back(idA);
                B.push_back(idA);
                S.push_back(1.0);
            }

            while (!qB.empty()) {
                int idB = qB.front(); qB.pop();
                pB.pop();
                A.push_back(idB);
                B.push_back(idB);
                S.push_back(1.0);
            }
        }

        int roll() {


	    // if ((int)A.size() == 0) {
		// // cerr << "ERROR:: A.size() == 0 in Alias Table" << endl;
		// // exit(0);
	    // }
            int x = rand() % ((int)A.size());
            double p = 1.0 * rand() / RAND_MAX;
            return p>S[x] ? A[x] : B[x];
        }

}AT[N];

void link(int u, int v)
{
    E[u].push_back(v);
}

int roll(int u, vector<int> path)
{

    vector<int> result;
    set_difference(E[u].begin(),E[u].end(),path.begin(),path.end(),inserter(result,result.begin()));
    int len=result.size();
    if(len==0)
        return -1;
    int num=rand()%len;
    return result[num];
}

void bfs(int S)
{
    queue<int> q;
    q.push(S);
    dis[S][S] = 1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        if (dis[S][u] > seq_len)
            return;
        for (int i = 0; i < (int)E[u].size(); i++)
        {
            int v = E[u][i];
            if (dis[S][v] == 0)
            {
                dis[S][v] = dis[S][u] + 1;
                q.push(v);
            }
        }
    }
    return;
}

int main(int argc, char *argv[])
{

    if (argc != 6)
    {
        cerr << "ERROR: Incorrect number of parameters. " << endl;
        return 0;
    }

    stringstream ss1, ss2;
    ss1.str("");
    ss1 << "data/";
    ss1 << argv[5];
    ss1 << "/";
    ss1 << argv[1];
    // ss1 << "_nsl";
    ss1 << ".in";

    ss2.str("");
    ss2 << "data/";
    ss2 << argv[5];
    ss2 << "/";
    // ss2 << "/data/syf/rw/";
    ss2 << argv[1];
    ss2 << "_";
    ss2 << argv[2];
    ss2 << "_";
    ss2 << argv[3];
//    ss2 << "_";
//    ss2 << "nsl";
    ss2 << "_merw.txt";
    
    num_of_walks = atoi(argv[2]);
    seq_len = atoi(argv[3]);
    start = atoi(argv[4]);

    cout << "File input: " << ss1.str().c_str() << endl;
    cout << "File output: " << ss2.str().c_str() << endl;

    cout << "aaaaa" << endl;

    freopen(ss1.str().c_str(), "r", stdin);
    freopen(ss2.str().c_str(), "w", stdout);
    srand(time(0));
    scanf("%d%d", &n, &m);

    cerr << argv[1] << ": " << n << endl;
    // fprintf(stderr,"add link...\n");
    for (int i = 1; i <= m; i++)
    {
        int u, v;
        // double p;
        scanf("%d%d", &u, &v);
        // scanf("%d%d%lf", &u, &v, &p);
        // if(p==0) continue;
        link(u, v);
    }
    // fprintf(stderr,"bfs...\n");
    //计算距离
    for (int i = 0; i < n; i++){
        // fprintf(stderr,"%d\n",i);
        bfs(i);
    }

    // fprintf(stderr,"add node...\n");
    for (int epoch = 0; epoch < 1; epoch++)
    {
        // for (int st = 4716; st < n; st++)
        for (int st = start; st < n; st++)
        {
            for (int i = 0; i < num_of_walks; i++)
            {
                int u = st;
                if ((int)E[u].size()==0){break;}
                else{
                    //printf("[");
                    //存结点
                    vector<int> path;
                    string s="";
                    int t = 0;
                    int flag = 0;
                    for (t; t < seq_len; t++)
                    {                    

                        //printf("%d", u);
                        s += to_string(u);
                        path.push_back(u);
                        // fprintf(stderr, "%d\n", u);
                        o[t] = dis[st][u];
                        //printf(" ");
                        s += " ";
                        int g = roll(u,path);
                        if(g==-1){
                            flag=1;
                            break;
                        }
                        u = g;
                    }
                    if(flag==1){
                        printf("%s",s.c_str());
                        fprintf(stderr,s.c_str());
                        for(t;t<seq_len;t++){
                            printf("-1 -1");
                            fprintf(stderr,"-1 -1 ");
                            if(t<seq_len-1) {
                                printf(" ");
                                fprintf(stderr," ");}
                                }
                    }
                    else{
                        printf("%s",s.c_str());
                        fprintf(stderr,s.c_str());
                    }
                
                    //存距离
                    for (int _ = 0; _ < seq_len; _++)
                    {   
                        if(flag==1) break;
                        printf("%d", o[_] - 1);
                        fprintf(stderr, "%d ", o[_] - 1);
                        if (_ != seq_len - 1)
                            printf(" ");
                    }

                    printf("\n");
                    fprintf(stderr, "\n");
                }
            }
        }
    }
    fclose(stdin);
    fclose(stdout);

    return 0;
}

extern "C" int* function(){
    ifstream inputfile("gene_40_4_merw.txt");
    char str[1000000];//N是定义的常数，目的是为了读取足够长的行
    int n=0;//用来表示说读/写文本的行数
    while(!inputfile.eof())
    {
    inputfile.getline(str, sizeof(str));//此处默认的终止标识符为‘\n’
    n++;
    }
    int row = n, col = 8;

    int myArray[row][col];
    ifstream file("gene_40_4_merw.txt");
    for (int r = 0; r < row; r++) //Outer loop for rows
        {
            for (int c = 0; c < col; c++) //inner loop for columns
            {
            file >> myArray[r][c];  //Take input from file and put into myArray
            }
        }
    int* b = new int[row*col+1];
    int cnt=0;
    for(int i = 0; i < row; i++)
        {
            for(int j = 0; j<col; j++)
            {
                b[cnt]=myArray[i][j];
                cnt++;
            }
        }
    b[row*col+1]=-1;
    return b;
}