/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of TextMining Project by
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source
along with  rest of the project. However any copy/redistribution,
including but not limited to compilation to binaries, must carry
this header in its entirety. A note must be made about the origin
of your copy.

TextMining is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils/Utils.hxx"

using namespace std;
using namespace Utils;

typedef std::map<string, double> TermFreq;

struct DocCount_t
{
    std::map<string, size_t> DocCountMap;
    std::mutex Mutex;
} DocCount;

struct DocInfo
{
    string Content;
    TermFreq DocCounts;
};
vector<DocInfo*> DocInfos;

struct SearchResult {
    double TFIDF;
    size_t offset;
    operator double() { return TFIDF; }
};

size_t makeStopWords(std::unordered_set<std::string>& StopWords)
{
    ifstream in("StopWords");
    while (in)
    {
        string word;
        getline(in, word);
        StopWords.insert(word);
    }
    return StopWords.size();
}

void MakeTF(DocInfo* info, std::unordered_set<std::string>& StopWords)
{
    const char* delims = " :\t";
    char* tok = strtok(&(info->Content[0]), delims);
    
    while ((tok = strtok(0, delims)) != nullptr)
        if(StopWords.find(tok) == StopWords.end())
            info->DocCounts[tok]++; // count for TF

    size_t mx = 0;
    for (auto& t : info->DocCounts)
        mx = std::max(mx, size_t(t.second));
       
    for (auto& t : info->DocCounts)
        t.second /= mx; // finally this is TF

    DocCount.Mutex.lock();
    
    for (auto& t : info->DocCounts)
        DocCount.DocCountMap[t.first]++;
    
    DocCount.Mutex.unlock();
}

void Monitor(bool& on)
{
    while (on)
    {
        cout << "\r" << DocInfos.size() << " Done!";
        std::this_thread::sleep_for(chrono::milliseconds(500));
    }
}

void SearchDocs(size_t thrdIdx, size_t maxThrds, const char* term, std::vector<SearchResult>& results)
{
    SearchResult ret = { Eps, size_t(-1) };
    for (size_t id = thrdIdx; id < DocInfos.size(); id += maxThrds)
    {
        auto found = DocInfos[id]->DocCounts.find(term);
        if (found != DocInfos[id]->DocCounts.end() && found->second > ret.TFIDF)
            ret = { found->second, id };
    }
    results[thrdIdx] = ret;
}

void Search(size_t searchParallelSize = 32)
{
    char onMore = 'n';
    do
    {
        string searchTerm = "";
        cout << "Enter search Term: ";
        cin >> searchTerm;

        std::vector<SearchResult> results(searchParallelSize);
        ThreadPool threads(searchParallelSize);

        auto start = chrono::steady_clock::now();
        for (size_t i = 0; i < searchParallelSize; ++i)
            threads.Launch(SearchDocs, i, searchParallelSize, searchTerm.c_str(), std::ref(results));
        threads.JoinAll();

        cout << "search done in " << Utils::TimeSince(start) << "s\n";

        std::remove(results.begin(), results.end(), Eps);
        if (results.size())
        {
            std::sort(results.begin(), results.end());
            cout << "Closest matching document: " << endl
                << " Doc#: " << results.back().offset
                << " TFIDF: " << results.back().TFIDF << "\n ====== \n"
                << DocInfos[results.back().offset]->Content
                << "\n ====== \n";
        }
        else
            cout << "Nothing found\n";
        

        cout << "One more? [y/n] ";
        cin >> onMore;
        cout << endl;

    } while (onMore == 'y');
}

int main(int argc, char** argv)
{
    ifstream in("data");
    if (!in) throw std::runtime_error("Failed to open file");

    std::unordered_set<std::string> StopWords;

    if (!makeStopWords(StopWords))
        std::runtime_error("Stop words could not be populated");

    ThreadPool threads(256);
    bool monitor = true;
    std::thread monitorThrd(Monitor, std::ref(monitor));

    auto start = std::chrono::steady_clock::now();
    while (in)
    {
        DocInfo* doc = new DocInfo();
        getline(in, doc->Content);

        DocInfos.push_back(doc);
        threads.Launch(MakeTF, doc, std::ref(StopWords));
    }

    threads.JoinAll();
    monitor = false; monitorThrd.join();
    cout << "Done in: " << Utils::TimeSince(start) << "s.\n";

    size_t NumDocs = DocInfos.size();

    for (auto& d : DocInfos)
        for (auto& dc : d->DocCounts)
            dc.second *= NumDocs / DocCount.DocCountMap[dc.first];

    Search();


    for (auto d : DocInfos)  delete d;
        

    return 0;
}
