from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import time
import sentiment_module as s

#consumer key, consumer secret, access token, access secret.
ckey="WqTon7Fv2PMEnmRFbPy7h6YgC"
csecret="JWRRBqfeHxI9UHJEevOT6OC0LWZDvxINl7qwG8adjWwRVtOkgB"
atoken="1300894698-F2d3OVwz4ToQ9CljHbiHt4i017rvTvcGYXo13S8"
asecret="Wb3a6uaS8ynyUAw3sKRA95cCKtNCpiy8I3wLV3dg9zl52"

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)

        tweet = all_data["text"]

        # time.sleep(0.3)
        sentiment_value = s.sentiment(tweet)
        print(tweet, sentiment_value)

        with open("twitter_out.txt", 'a') as f:
            f.write(sentiment_value)
            f.write('\n')

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["India"])
