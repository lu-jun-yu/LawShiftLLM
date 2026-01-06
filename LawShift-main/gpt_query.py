import requests
import json
import http.client


def gpt_api(query, temperature=0.3, model="gpt-3.5-turbo-0125"):
    try:
        header = {
            "Authorization": "some keys",
            "Content-Type": "application/json"
        }
        body = {
            "model": model,
            # "model": "gpt-4",
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature
        }
        response = requests.post("a url", headers=header,
                                 data=str(body).replace("'", "\"").encode("utf-8"))
        response = json.loads(response.content.decode("utf-8"))
        output = response["choices"][0]["message"]["content"]
        # print(output)
    except:
        output = "ERROR"
    return output


def gpt_api_chat(sys, user, temperature=0.3, model="gpt-3.5-turbo-0125"):
    try:
        conn = http.client.HTTPSConnection("a url")
        payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": sys
            },
            {
                "role": "user",
                "content": user
            }
        ]
        })
        headers = {
        'Authorization': 'some keys',
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        # print(data.decode("utf-8"))
        output = json.loads(data.decode("utf-8"))["choices"][0]["message"]["content"].replace("\n", "")
    except:
        output = "ERROR"
    return output
