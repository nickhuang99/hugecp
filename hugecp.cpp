#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <map>
#include <string>
#include <getopt.h>

using namespace std;

extern int errno;

#ifdef HUGEPAGE_SIZE_1G
#define HUGE_PAGE_SIZE 1073741824
//  sanity test
static_assert(sizeof(off_t) == 8);
static_assert(HUGE_PAGE_SIZE == 1073741824);
#else
#define HUGE_PAGE_SIZE 2097152
#endif

static char buffer[HUGE_PAGE_SIZE];
static int64_t pageSize  = HUGE_PAGE_SIZE;
static off_t srcSize = 0;
static int64_t tgtSize = 0;
static off_t totalCopySize = 0;

void update_progress(int progress) {
    int bar_length = 40; // Modify this to change the bar's length
    int filled_length = (int)(bar_length * progress / 100.0);
    char bar[bar_length + 1]; // +1 for the null terminator
    for (int i = 0; i < bar_length; i++) {
        if (i < filled_length) {
            bar[i] = '=';
        } else {
            bar[i] = '-';
        }
    }
    bar[bar_length] = '\0'; // Null-terminate the string
    printf("\r[%s] %d%%", bar, progress);
    fflush(stdout); // Ensure output is written immediately
}

bool copyOneFile(const char *fileName, off_t fileSize, off_t pageSize, char *ptr) {
    bool result = true;
    int src_fd;
    src_fd = open(fileName, O_RDONLY);
    if (src_fd == -1) {
        printf("source file %s cannot be opened! %s\n", fileName, strerror(errno));
        return false;
    }
    int64_t pageNumber = (fileSize + pageSize - 1) / pageSize;
    int64_t copySize = 0;
    for (int64_t i = 0; i < pageNumber; i++) {
        int size;
        if ((size = read(src_fd, buffer, pageSize)) == pageSize ||
            (size > 0 && i == pageNumber - 1)) {
            memcpy(ptr, buffer, size);
            ptr += pageSize; // last page no need to worry
            copySize += size;
            totalCopySize += size;
            update_progress(totalCopySize * 100 / tgtSize);
        } else {
            if (size == -1) {
                printf("read source file %s failed with error %s\n", fileName, strerror(errno));
                result = false;
                break;
            } else {
                // this is complicated situation, it require you to do a repeating call
                // let's do it later
                printf("returned reading number is not expected as %lu", pageSize);
                result = false;
                break;
            }
        }
    }
    // printf("file %s size %lu copied %lu %s\n", fileName, fileSize, copySize, result?"success":"failure");
    return result;
}

bool openDirectory(const string& dirName, map<string, off_t>& filesInfo, off_t& totalSize) {
    bool result = true;

    DIR* dir = opendir(dirName.c_str());
    
    if (!dir) {
        printf("directory %s cannot be opened %s\n", dirName.c_str(), strerror(errno));
        return false;
    }

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {            
        // filer special entry
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        if (ent->d_type != DT_REG) continue; // don't support recursive
        struct stat st;
        string fileName = dirName + "/" + ent->d_name;
        if (stat(fileName.c_str(), &st) != 0) {
            printf("stat file %s failed %s\n", fileName.c_str(), strerror(errno));
            result = false;
            break;
        }
        filesInfo.insert(make_pair(fileName, st.st_size));
        totalSize += st.st_size;   
    }
    closedir(dir);
    return result;
}

int main(int argc, char **argv) {
    bool verbose_flag = false;
    struct option long_options[] = {
        {"verbose", no_argument, 0, 'v'},
        {"source", required_argument, 0, 'i'},
        {"target", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0} // Required terminator
    };
    const char* help = "[--verbose] <-i sourceFilename|sourceDirectory > <-o targetFilename> [--help]";

     // Check for no arguments or just program name
     if (argc <= 1) {
        fprintf(stderr, "Error: -o and -i options are required.\n");
        printf("Usage: %s %s\n", argv[0], help);
        return 1;
    }
    char* srcNamePtr = nullptr, *tgtNamePtr = nullptr;
    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "hvo:i:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'v':
                verbose_flag = true;
                break;
            case 'i':
                srcNamePtr = optarg;
                break;
            case 'o':
                tgtNamePtr = optarg;
                break;
            case 'h':
                printf("Usage: %s %s\n", argv[0], help);
                return 0;
            case '?':
                /* getopt_long already printed an error message. */
                return 1;
            default:
                abort();
        }
    }

    struct stat st;

    if (stat(srcNamePtr, &st) != 0) {
        printf("source file %s is not valid file: %s\n", srcNamePtr, strerror(errno));
        return -2;
    }
    map<string, off_t> filesInfo;
    if (S_ISREG(st.st_mode)) {
        printf("copy mode for single model file %s\n", srcNamePtr);
        srcSize = st.st_size;
        filesInfo.insert(make_pair(srcNamePtr, st.st_size));
    } else if (S_ISDIR(st.st_mode)) {
        printf("copy mode for directory model file %s\n", srcNamePtr);
        if (!openDirectory(srcNamePtr, filesInfo, srcSize)) {
            printf("collect source directory files info failed!\n");
            return -6;
        }
    }
    // target size for mmap must be aligned with pageSize;
    tgtSize = (srcSize + pageSize - 1) / pageSize * pageSize;
    int tgtFd;  
    tgtFd = open(tgtNamePtr, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (tgtFd == -1) {
        printf("target file %s cannot be opened! %s\n", tgtNamePtr, strerror(errno));
        return -6;
    }
    printf("prepare to mmap target size %lu for source size  %lu\n", tgtSize, srcSize);
    void *ptr = mmap(NULL, tgtSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, tgtFd, 0);
    if (ptr == MAP_FAILED) {
        printf("mmap target file %s failed %s", tgtNamePtr, strerror(errno));
        return -7;
    }
    close(tgtFd); // immediately close is better
    char *tgtPtr = (char *)ptr;

    bool result = true;
    printf("prepare to concatenate model files at following order:\n");
    for (auto it = filesInfo.begin(); it != filesInfo.end(); it ++) {
        printf("name: %s size: %lu\n", it->first.c_str(), it->second);
    } 
    // we have to assume all model files's name must be alphabetical ordered
    for (auto it = filesInfo.begin(); it != filesInfo.end(); it ++) {
        if (!copyOneFile(it->first.c_str(), it->second, pageSize, tgtPtr)) {
            result = false;
            break;
        }
        tgtPtr += it->second;
        // update_progress(totalCopySize * 100 / tgtSize);
    }
   
    printf("\n%s copy from %s to target %s of total size %lu finished %ld\n", result?"Succeed":"Failed", 
        srcNamePtr, tgtNamePtr, srcSize, totalCopySize);
    munmap(ptr, tgtSize);  
    return 0;
}
