# hugecp
**Hugecp a Huge Leap**
* **What**
Hugecp is a simple tool to **copy** your file to `[hugetlbfs](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)`
It is efficient when model files are huge, for example, DeepSeek-r1:671b can be as big as 404G and it requires memory 
more than 377G to load into memory. And such huge memory creates a lot of "pages" if using **mmap**. So, in order to 
reduce page lookup table size and cache misses, hugepage can contribute.
* **Why**
You want to take advantage to **preload** your model file into **memory**. In this case, it is not **mlock** in ollama. 
Instead it uses **hugetlbfs**, a RAM-backed filesystem.

* **How**
    ```
        1.  Make sure you setup correct kernel boot parameter to allow appropriate `**hugepagesz**`.
        In case of "1G" hugepage size, you need to setup these three parameter at boot for kernel:
        default_hugepagesz=1G hugepagesz=1G hugepages=YourHugePageNumber
        In case of Ubuntu, the best editing place is to edit /etc/default/grub and edit
        "GRUB_CMDLINE_LINUX_DEFAULT=" to add these parameter. Then run "sudo update-grub" and reboot.
        **BECAREFUL**： choose number of huge page properly because you may consume all your RAM. 
        2.  Either allocate appropriate number of hugepages when boot or dynamically using sysctl.
        In case of 2M hugepage size, you can modify page number directly by sysctl. i.e.
        sudo sysctl -w vm.nr_hugepages=YourHugePageNumber
        verifying by cat /proc/meminfo | grep -i huge to look for "HugePages_Total".
        3.  Mount hugetlbfs. For example, you wan tto mount point at /mnt/hugepages
        sudo mount -t hugetlbfs none /mnt/hugepages
        You can also add option to allow yourself to be owner of mount point with 
        sudo mount -t hugetlbfs -o uid=$(id -u),gid=$(id -g),rw none /mnt/hugepages
        4.  Now run this tool to **copy** your model file to mount point. 
    ```

* **When**
Using this tool only when you have sufficient memory. i.e. I have no GPU, but huge memory like 1.5T


* **Updates**
    ```
        -  Mar 24, 2025 now accept source as directory name which means all models under directory will be concatenated.
        -  Mar 26, 2025 add argument parsing option
        -  Apr 8, 2025 add another tool q8_bf16.cpp. This is for converting `[fp8_cast_bf16.py](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main/inference)` DeepSeek-R1 fp8 to bf16 dequantization with pure CPU. The provided DeepSeek python script requires GPU with very large GPU memory which is not available for me.
    
    ```

