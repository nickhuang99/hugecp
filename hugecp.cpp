
#include <sys/mman.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

extern int errno;

#define HUGE_PAGE_SIZE 2097152

static_assert(sizeof(off_t) == 8);

int main(int argc, char**argv)
{        
        if (argc != 3) {
                printf("usage: %s srcfile tgtfile\n", argv[0]);
                return -1;
        }
        struct stat st;

        if (stat(argv[1], &st) != 0) {
                printf("source file %s is not valid file: %s\n", argv[1], strerror(errno));
                return -2;
        }
        off_t srcSize = st.st_size;
        int64_t pageNumber = (srcSize + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE;
        int64_t tgtSize = pageNumber * HUGE_PAGE_SIZE;
        int src_fd, tgt_fd;
        src_fd = open(argv[1], O_RDONLY);
        if (src_fd == -1) {
                printf("source file %s cannot be opened! %s\n", argv[1], strerror(errno));
                return -5;
        } 
        tgt_fd = open(argv[2], O_CREAT|O_RDWR|O_EXCL, 0666);
        if (tgt_fd == -1) {
                printf("target file %s cannot be opened! %s\n", argv[2], strerror(errno));
                close(src_fd);
                return -6;
        }
         
        void* ptr = mmap(NULL, tgtSize, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_HUGETLB, tgt_fd, 0);
        
        if (ptr == MAP_FAILED) {                
                printf("mmap target file %s failed %s", argv[2], strerror(errno));
                close(src_fd);                
                return -7;
        }        
        char* tgt_ptr = (char*)ptr;
        char buffer[HUGE_PAGE_SIZE];
        for (int64_t i = 0; i < pageNumber; i ++) {
                int size;
                if ((size = read(src_fd, buffer, HUGE_PAGE_SIZE)) == HUGE_PAGE_SIZE 
                        || (size > 0 && i == pageNumber - 1)) {
                        memcpy(tgt_ptr, buffer, size);
                        tgt_ptr += HUGE_PAGE_SIZE; // last page no need to worry
                } else {
                        if (size == -1) {
                                printf("read source file %s failed with error %s\n", argv[1], strerror(errno));
                                break;
                        } else {
                                // this is complicated situation, it require you to do a repeating call
                                // let's do it later
                                printf("returned reading number is not expected as %d", HUGE_PAGE_SIZE);
                                break;
                        }
                }
        }
        printf("copy from %s to target %s finished for file size %ld\n", argv[1], argv[2], tgtSize);
        munmap(ptr, tgtSize); 
        close(tgt_fd); // immediately close is better
        close(src_fd); 
        return 0;              
}
