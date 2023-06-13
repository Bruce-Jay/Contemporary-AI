import requests

# Replace with your GitHub token
token = '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'

# Replace with the owner and repo you're interested in
owner = 'streamich'
repo = 'react-use'

headers = {'Authorization': f'token {token}'}

response = requests.get(f'https://api.github.com/repos/{owner}/{repo}', headers=headers)

if response.status_code == 200:
    data = response.json()
    # print(f"Forks: {data['forks_count']}")
    # print(f"Stars: {data['stargazers_count']}")
    print(data)
else:
    print(f"Error: {response.status_code}")

