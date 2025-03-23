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

using namespace std;

extern int errno;

#ifndef HUGE_PAGE_SIZE
#define HUGE_PAGE_SIZE 1073741824 // 2097152
#endif

//  sanity test
static_assert(sizeof(off_t) == 8);
static_assert(HUGE_PAGE_SIZE == 1073741824);

static char buffer[HUGE_PAGE_SIZE];

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
            update_progress(i * 100 / pageNumber);
            copySize += size;
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
    printf("file %s size %lu copied %lu %s\n", fileName, fileSize, copySize, result?"success":"failure");
    return result;
}

bool openDirectory(const char* dirName, map<string, off_t>& filesInfo, off_t& totalSize) {
    bool result = true;
    totalSize = 0;

    DIR* dir = opendir(dirName);
    
    if (!dir) {
        printf("directory %s cannot be opened %s\n", dirName, strerror(errno));
        return false;
    }

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {            
        // filer special entry
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        if (ent->d_type != DT_REG) continue; // don't support recursive
        struct stat st;
        if (stat(ent->d_name, &st) != 0) {
            printf("stat file %s failed %s\n", ent->d_name, strerror(errno));
            result = false;
            break;
        }
        filesInfo.insert(make_pair(ent->d_name, st.st_size));
        totalSize += st.st_size;   
    }
    closedir(dir);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("usage: %s <srcFile|srcDirectory> tgtFile\n", argv[0]);
        return -1;
    }
    struct stat st;

    if (stat(argv[1], &st) != 0) {
        printf("source file %s is not valid file: %s\n", argv[1], strerror(errno));
        return -2;
    }
    off_t srcSize = 0;
    map<string, off_t> filesInfo;
    if (st.st_mode == S_IFREG) {
        srcSize = st.st_size;
        filesInfo.insert(make_pair(argv[1], st.st_size));
    } else if (st.st_mode == S_IFDIR) {
        if (!openDirectory(argv[1], filesInfo, srcSize)) {
            printf("collect source directory files info failed!\n");
            return -6;
        }
    }
    int64_t pageSize = HUGE_PAGE_SIZE;
    // target size for mmap must be aligned with pageSize;
    int64_t tgtSize = (srcSize + pageSize - 1) / pageSize * pageSize;
    int tgtFd;  
    tgtFd = open(argv[2], O_CREAT | O_RDWR | O_EXCL, 0666);
    if (tgtFd == -1) {
        printf("target file %s cannot be opened! %s\n", argv[2], strerror(errno));
        return -6;
    }
    void *ptr = mmap(NULL, tgtSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_HUGETLB, tgtFd, 0);
    if (ptr == MAP_FAILED) {
        printf("mmap target file %s failed %s", argv[2], strerror(errno));
        return -7;
    }
    close(tgtFd); // immediately close is better
    char *tgtPtr = (char *)ptr;
    off_t totalCopySize = 0;
    bool result = true;
    // we have to assume all model files's name must be alphabetical ordered
    for (auto it = filesInfo.begin(); it != filesInfo.end(); it ++) {
        if (!copyOneFile(it->first.c_str(), it->second, pageSize, tgtPtr)) {
            result = false;
            break;
        }
        tgtPtr += it->second;
        totalCopySize += it->second;
        update_progress(totalCopySize * 100 / tgtSize);
    }
   
    printf("\n%s copy from %s to target %s of total size %lu finished %ld\n", result?"Succeed":"Failed", 
        argv[1], argv[2], srcSize, totalCopySize);
    munmap(ptr, tgtSize);  
    return 0;
}
