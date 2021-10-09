http://qiuxuewei.com/post/git-he-bing-liang-ge-bu-tong-de-cang-ku/

# How to merge two different git repositories?



#### 1. 下载需要进行合并的仓库 B

```
git clone git@e.coding.net:qxuewei/notebook/notebook.git
```

#### 2. 添加需要被合并的远程仓库 A

```
git remote add base git@github.com:qxuewei/notebook.git
```

将 base 作为远程仓库，添加到 本地仓库(origin)中，设置别名为 base(自定义，为了方便与本地仓库origin作区分)

此时使用 `git remote` 查看所有远程仓库将看到两个 一个本地默认仓库origin 另外一个我们新增的 base

#### 3. 把base远程仓库（A）中数据抓取到本仓库（B）

```
git fetch base
```

第2步 `git remote add xxx` 我们仅仅是新增了远程仓库的引用，这一步真正将远程仓库的数据抓取到本地，准备后续的更新。

#### 4. 基于base仓库的master分支，新建一个分支，并切换到该分支，命名为 "githubB"

```
git checkout -b githubB base/master
```

此时我们的仓库B就有了一个基于仓库A内容的分支 "githubB"，后续我们将 "githubB" 分支代码合并到master就可以了。

此时使用 `git branch` 查看所有分支

#### 5. 我们切换到需要合并的分支 master

```
git checkout master
```

第 4 步我们创建了即将被合并分支 "githubB" ，默认是在当前分支上的，所以我们需要切换回我们的目标分支。

#### 6. 合并

```
git merge githubB --allow-unrelated-histories
```

如果不加 `--allow-unrelated-histories` 关键字会报错

```
fatal: refusing to merge unrelated histories
```

如果在流程中报上述错误加该关键词`--allow-unrelated-histories`即可。

合并过程中可能会遇到各种冲突，如果有冲突解决就可以了。
