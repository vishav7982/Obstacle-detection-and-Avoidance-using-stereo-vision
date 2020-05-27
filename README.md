# Obstacle-detection-and-Avoidance

This project is done Under the Guidance of Dr. S.N Omkar (chief scientist IISC bangalore) 
It is part of Our Research Intern 
We were assigned a task of avoiding Obstacle in front of our drone With as minimum computation as possible
We tried many algorithms but didnt get good results.
So, finally we came up with ORB matching Algorithm and with the help of ORB we can calculate disparity b/w pixels in left and right Image
and With the help of disparity we can calculate distance of that object from our drone.
For Minimum Computation, we divided our frame into region of Interest and then we are calculating disparity only in our region of Interest.


With this Approach, we got around 55 fps in a low end Laptop.
