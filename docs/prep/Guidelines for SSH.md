# Guidelines for SSH
Once can access the `ssh` command through the following process:

1. Copy and paste the following configuration file in the `C:/Users/{USER}` home directory with the filename `.wezterm.lua`. (this is located in `assets/.wezterm.lua`). 

2. Now in wezterm, we do the following commands.
```bash
> Ubuntu #enters wsl ubuntu environment
> ssh-keygen #generates ssh-key
> ssh -p 31018 neuralode@147.46.92.61 -i ~/.ssh/id_rsa #enter ssh with private key
```
3. After doing Step 2, you should see the following 
``` bash
root ssh at ~
>
```
4. You switch to the `neuralode` user by the `su` command. 

```bash
> su neuralode
> cd ~
```
5. Now you see the available `tmux` sessions by
```bash
> tmux ls
> tmux a -t {session_name}
```
where session name is the name of the session name. 

6.  `pip` and other dependencies can be activated by
```bash
> conda --info envs #lists all available environments
> conda activate catbase_main
```



