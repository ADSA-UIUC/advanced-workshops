# Git Crash Course

## Introduction

Git is the defacto version control system for developers. 
It's used to mange versions of your work, and collaborate in teams at scale. 
GitHub is a hosting platform for Git projects. 
There are other options (i.e. GitLab, custom, etc.), but we're going to use GitHub 
because it is the most popular. Over 40+ million projects are hosted on GitHub including
the Linux Kernel, the Python Programming Language, the VS Code text editor, etc. In short, 
Git and GitHub are meant for collaboration on both closed and open-source projects.

## Getting Started

1.) Install Git and create a free GitHub account

2.) On the GitHub website create a new repository, give it any name, and continue

### Cloning a repository 

This creates a local copy of the repository on your computer

1.) Copy the url to the repository

2.) In the terminal (Git Bash if on Windows) type `git clone repository-url` where repository-url is the above link

3.) Run the command `ls` to list the folders in your current directory. You should now see a folder for the repository.

### Branching

Whenever you are collaborating on the same team, you each want to work on different *branches*.
A branch isolates your work from someone elses, and then when you are finished your work can get
merged back together. If you do not branch, you may end up getting merge conflicts which occurs when 
your changes conflict with someone elses in the same branch.

1.) Go into the folder we just created `cd folder-name`. cd stands for change directory

2.) Make a new branch by running the command `git checkout -b new-branch-name`. The `-b` flag tells Git to create a 
new branch. If a branch already exists that you want to modify you would exclude that flag

3.) New branch only: To push (deploy) the branch, run `git push --set-upstream origin new-branch-name`

### Committing Your Changes

Whenever you make changes and want to save your work, you want to add-commit-push it. The `add` command stages the files to commit.
The `commit` command creates a version from the staged files, but the version is still local. The `push` command then pushes those 
changes to the public repository (i.e. GitHub). Make sure you are in the base directory of the project when doing this.

1.) `git add filename` will stage a specific file. `git add folder-name` will stage an entire folder. i.e. `git add .` will stage all 
the changes from the current directory. Avoid the command `git add *` as this can result in unwanted/unexpected problems if you don't 
know what it does.

2.) `git commit -m "your commit message here". The `-m` flag stands for message. You want to leave *good* commit messages so people
know what your changes are. 

Bad commit message: **"Updates"**, **"wtf"**, **"idek"**, **"jlshgslkd"**

Good commit messages: **"Fixed memory leaks in estimator.c"**

3.) Run the command `git push` to push all of the commits to GitHub. You should now see the changes to your branch on the repository's 
website.

### Pulling Others Changes 

When collaborating on a project, it is best to pull any changes as soon as you start editing each time. This will help to avoid merge 
conflicts, and keep your branch up to date.

`git pull` will pull any changes in the current branch 

If you want to pull changes from another branch into your branch, then you would want to merge their branch into yours. 
See the section on merging.

### More on Staging

- Before running the `commit` command, you can view the files that are modified, unstaged, or staged via the command `git status`.

- To unstage changes to a file, run the command `git rm filename`

#### Ignoring Files

To explicitly ignore changes to a file (i.e. passwords, environment config, etc.) create a .gitignore file that lists 
all the files you don't want to commit. 

Sample .gitignore file:
```
filename-to-ignore.txt 
passwords.json
folder-name-to-ignore 
folder1/folder2/script-to-ignore.py 
```

### Merging Branches 

When you are done making changes and want to merge your changes with another branch (i.e. `master`), then do the following

1.) add-commit-push all the changes you have made 

2.) Switch to the branch you want to *merge into* i.e. `git checkout master` (no `-b` flag since the `master` branch already exists)

3.) Pull any changes that were made `git pull`

3.) Merge in your branch `git merge branch-name`

If you want to see a list of all the branches, run `git branch -a`. The `-a` flag is for *all* branches 

#### Merge Conflicts 

Merge conflicts occur when your changes conflict or overlap with someone elses. When these occur, open up the files that have conflicts 
and fix the merge conflict by choosing whose lines of code to keep, and whose to delete 

Example of what a file looks like after a merge conflict: 
```
<<<<<<< HEAD
this is some content to mess with
content to append
=======
totally different content to merge later
>>>>>>> new_branch_to_merge_later
```

To resolve this, you would delete all the lines that you don't want to keep. i.e. we want to keep the second option of changes 
which are what we had made 

```
totally different content to merge later
```

Merge conflicts can and will happen, but they are nothing to be scared of! When they do happen there are a lot of good 
internet resources about how to resolve them.

## Best Practices

- Commit and push often. You should make small, incremental changes that can stand on their own (i.e. don't break everything)
- When merging into a base repository (i.e. master) your code **should probably work**. Do not introduce breaking changes into 
other peoples working repository
- Avoid flags such as `-f` or "force" unless you absolutely know what you are doing
- Avoid rebasing or squashing commits unless you absolutely know what you are doing 
- Use good commit messages so people know what your changes are 
- Only push files you want others to see
- ***Absolutely do not push any database/account/etc. passwords***, instead store these in a separate file (i.e. .env) and add that 
secret file to the .gitignore 
- When in doubt, ask for help or look on Stack Overflow
