complexity of hough transform:
------------------------------

n: dim of param space (eg. 2 for a line)
k: number of bins for each param
v: number of voters

complexity:

for each v:
  for each param:
    if param is suitable:
      acc[param] += 1

=> O(vk^n) or O(vk^n-1) in the case of the line (beacause last param can be obtained from the others)


interest of using the gradient for hough circle:
------------------------------------------------

using the gradient give information about theta => 1 dimension gain : O(vk^n-2)

