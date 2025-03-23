# hugecp
**Hugecp a Huge Leap**
* **What**
Hugecp is a simple tool to **copy** your file to `[hugetlbfs](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)`

* **Why**
You want to take advantage to **preload** your model file into **memory**. In this case, it is not **mlock** in ollama. 
Instead it uses **hugetlbfs**, a RAM-backed filesystem.

* **How**
    ```
        1.  Make sure you setup correct kernel boot parameter to allow appropriate `**hugepagesz**`.
        2.  Either allocate appropriate number of hugepages when boot or dynamically using sysctl.
        3.  Mount hugetlbfs
        4.  Now run this tool to **copy** your model file to mount point. 
    ```

* **When**
Using this tool only when you have sufficient memory. i.e. I have no GPU, but huge memory like 1.5T


* **Updates**
    ```
        -  Mar 24, 2025 now accept source as directory name which means all models under directory will be concatenated.
    
    ```

