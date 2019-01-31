#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <fstream>
#include "utils.h"


const char *g_usage[] = {
	_TEE_JSON_CONFIG_,
	_TEE_DATA_TEST_PATH_,
    _TEE_LABEL_NAME_,
	_TEE_FILE_LIST_
};

static std::string GetDefaultModelPath(void) {
	char path[512] = { 0 };
#ifdef _WIN32
	GetModuleFileName(0, path, 512);
	char *tail = path + strlen(path);
	while (tail != path) {
		if (*tail == '\\' || *tail == '/') break;
		else --tail;
	}
	*tail = 0;
#else
	getcwd(path, 512);
#endif
	return std::string(path) + "/model";
}

std::map<std::string, std::string> ParseArgs(int argc, char *argv[]) {
    std::map<std::string, std::string> cmdArgs;
    for (size_t i = 0; i < sizeof(g_usage) / sizeof(char *); ++i) {
        cmdArgs[g_usage[i]] = std::string();
    }
    for (int i = 1; i < argc; i += 2) {
        if (cmdArgs.find(argv[i]) != cmdArgs.end()) {
            cmdArgs[argv[i]] = argv[i + 1];
        }
    }
	if (cmdArgs[_TEE_DATA_TEST_PATH_].empty()) { cmdArgs[_TEE_DATA_TEST_PATH_] = GetDefaultModelPath(); };
    if (cmdArgs[_TEE_DATA_TEST_PATH_].back() != '/' && cmdArgs[_TEE_DATA_TEST_PATH_].back() != '\\') cmdArgs[_TEE_DATA_TEST_PATH_] += '/';
    return cmdArgs;
}



void printfUsage(void) {
	printf("\nParameter Error. Usage: \n\
-------- Inference Engine ---------\n\
*   dataTestPath   c_string(default path is ./Model)    \n\
*   configFile   c_string    \n\
*   labelName   c_string    \n\
*   fileList    c_string    \n\
-----------------------------------\n");
}

static std::string _NXGetLine2(std::ifstream &fIn) {
	std::string line;
	getline(fIn, line);
	if (line.empty()) return line;
	while (isspace(line.back())) {
		line.pop_back();
	}
	return line;
}
void _GetNameList2(std::string const &name, std::vector<std::string> &vtName) {
	std::ifstream fIn(name);
	if (!fIn.is_open()) return;
	while (true) {
		std::string line = _NXGetLine2(fIn);
		if (line.empty()) break;
		vtName.push_back(line);
	}
}

std::string _getPath(std::string const &name) {
	std::string::size_type pos1 = name.rfind('\\');
	std::string::size_type pos2 = name.rfind('/');
	std::string::size_type pos = std::string::npos;
	if (pos1 == std::string::npos) pos = pos2;
	else if (pos2 == std::string::npos) pos = pos1;
	else pos = pos1 > pos2 ? pos1 : pos2;
	if (pos == std::string::npos) return std::string();
	return std::string(name.begin(), name.begin() + pos);
}

void _GetNameList(std::string const &name, std::vector<std::string> &vtName) {
	std::ifstream fIn(name);
	if (!fIn.is_open()) return;
	std::string npath = _getPath(name);
	while (true) {
		std::string line = _NXGetLine2(fIn);
		if (line.empty()) break;
		vtName.push_back(npath + "/" + line);
	}
}

std::string readFileIntoString(const char * filename)
{
	std::ifstream ifile(filename);
	std::ostringstream buf;
	char ch;
	while (buf&&ifile.get(ch))
		buf.put(ch);
	return buf.str();
}


