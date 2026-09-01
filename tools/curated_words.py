#!/usr/bin/env python3
"""Curated append-only additions: chat slang for en_words.txt and given names for
en_caps.txt. Idempotent; run from the repo root.

Slang admission is by hand. Appended ids sit past the trained range, so the network
scores them as UNK and they rank last as suggestions: the effect is purely that the
known-word gate stops "correcting" them. Same append-only contract as expand_vocab.py
(existing line numbers never move).

Every curated name gets a caps entry, including the ones with a live lowercase
reading (mark, jack, holly): a stray capital in "mark my words" is a smaller wrong
than leaving someone's name lowercase, so the name reading wins wholesale. When the
vocabulary grows cased ids, the net can learn to hold the capital back where the
common-noun reading fits the context. Names absent from en_words.txt are appended
there too, else the gate would autocorrect them before casing ever ran.
"""

import sys

VOCAB = "app/src/main/assets/en_words.txt"
CAPS = "app/src/main/assets/en_caps.txt"
CAPS_MARK = "# Curated given names (tools/curated_words.py): the name reading wins over any lowercase one."
ID_CAP = 128000  # matches expand_vocab.py; ids past 16 bits get no bigrams, UNK in the net

SLANG = """ahaha ahahaha aww awww bday bestie besties bff bffr bffs brb bruh bruv
bussin deadass defo ehh erm eww ewww ez ffs finna flexin fomo fyi gg ghosting gimme
gotcha grl gtg hbu heh hella highkey hmmm hmu idc idgaf ikr ily ilysm imma innit irl
jk kk lemme lmfao lmk lolz lowkey meh nahh ngl ngmi nocap np nvm obv obvi ofc ohh
ohhh omfg omw oof oooh oopsie pfft pls plz prolly psst rizz rly rofl sheesh shh shhh
smh sorta stfu tf thx tmrw ttyl tysm ughh uhh uhhh umm ummm vibin vibing wbu welp
whatevs wyd y'all yass yasss yayy yeet yeppers yikes yolo""".split()

NAMES = """jason aaron adam alan albert alex alexander alexandra alexis alice alicia
allison alyssa amanda amber amelia amy andrea andrew angela ann anna anne annie
anthony antonio ashley austin barbara benjamin bethany betty beverly bobby bradley
brandon brenda brendan brett brian brianna brittany brooke bruce bryan caleb cameron
carl carlos carmen caroline carrie casey catherine cathy charles charlotte chelsea
cheryl chris christian christina christine christopher cindy claire clara clarence
claudia cody colin connor courtney craig crystal cynthia dale damian dana daniel
danielle danny darren dave david deborah debra denise dennis derek desmond diana
diane dominic donald donna doris dorothy douglas dustin dylan edward eleanor elena
elias elijah elizabeth ella ellen emily emma eric erica erik erin ethan eugene eva
evan evelyn felix fiona frances francis gabriel gabriella gareth garrett gary
gavin geoffrey george gerald gina gloria gordon graham greg gregory gretchen
hannah harold harry hayley heather heidi helen henry holly howard hunter ian irene
isaac isabel isabella isabelle jack jackie jacob jacqueline jake james jamie jane
janet janice jared jasmine jay jayden jean jeff jeffrey jenna jennifer jenny
jeremiah jeremy jerome jerry jesse jessica jill jim jimmy joan joanna joanne jodie
joe joel joey john johnny jonathan jordan jorge jose joseph josh joshua joyce juan
judith judy julia julian julie justin kaitlyn karen kate katelyn katherine kathleen
kathryn kathy katie kayla keith kelly kelsey ken kendra kenneth kevin kim kimberly
kirsten kristen kristin kurt kyle larry laura lauren laurie lawrence leah lee leo
leonard leslie liam lillian lily linda lindsay lisa logan lois lori lucas lucy luis
luke lydia lynn madeline madison maggie marcus margaret maria marie marilyn marissa
mark marsha martha martin marvin mary mason matt matthew maureen megan melanie
melissa meredith michael michelle miguel mike miranda mitchell molly monica morgan
nancy naomi natalie nathan nathaniel neil nicholas nick nicole nina noah nolan
norman oliver olivia oscar owen pamela patricia patrick paul paula peggy peter
philip phillip phoebe phyllis rachel ralph randy raymond rebecca regina renee
richard rick ricky riley rita robert roberto robin rodney roger ronald rosa ross
roy russell ruth ryan sabrina sally sam samantha samuel sandra sara sarah scott
sean sebastian seth shannon sharon shawn sheila shelby shirley sidney simon sophia
sophie spencer stacy stanley stephanie stephen steve steven stuart susan suzanne
sydney sylvia tanya tara taylor ted teresa terry thomas tiffany tim timothy tina
toby todd tom tommy tony tracy travis trevor tristan tyler valerie vanessa veronica
victor victoria vincent virginia vivian walter wayne wendy wesley whitney william
xavier yvonne zachary zoe""".split()

# US states, major US and world cities, and the distinctive components of multi-word
# places (york, angeles, hampshire — never new). Held out: severe collisions where the
# lowercase reading is the primary chat use, not an occasional one (turkey the bird,
# jersey the shirt, buffalo the wings); those wait for cased vocab ids like the names.
PLACES = """alabama alaska arizona arkansas california colorado connecticut delaware
florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine
maryland massachusetts michigan minnesota mississippi missouri montana nebraska
nevada hampshire mexico york carolina dakota ohio oklahoma oregon pennsylvania
rhode tennessee texas utah vermont virginia washington wisconsin wyoming
angeles francisco diego antonio jose orleans juan vegas
dallas denver atlanta houston phoenix philadelphia philly sacramento portland
nashville memphis orlando tampa tucson albuquerque omaha tulsa wichita boise
spokane tacoma fresno oakland berkeley pasadena anaheim cleveland cincinnati
columbus detroit pittsburgh baltimore indianapolis milwaukee minneapolis
louisville raleigh richmond jacksonville honolulu anchorage reno brooklyn bronx
manhattan hollywood chicago boston seattle miami
bangkok phuket pattaya krabi melbourne toronto vancouver montreal dublin madrid
barcelona lisbon amsterdam berlin rome milan munich vienna prague budapest warsaw
oslo stockholm helsinki copenhagen athens cairo dubai delhi mumbai singapore seoul
beijing shanghai taipei manila jakarta hanoi saigon kyoto osaka""".split()

PLACE_HOLDOUTS = set("turkey nice mobile orange reading buffalo jersey bath split mesa normal surprise industry queens".split())

# Modern tech, AI, and everyday-medical vocabulary the frequency corpus lags on.
TECH = """qwen chatgpt gpt llm llms lora rag gguf agentic multimodal tokenizer
tokenizers finetune finetuned finetuning hallucinate hallucinated hallucination
hallucinations deepseek mistral ollama anthropic openai copilot midjourney
kubernetes devops fullstack oauth webhook webhooks microservice microservices
serverless postgres postgresql sqlite graphql grpc json yaml toml regex regexes
repo repos monorepo github gitlab rebase refactor refactored refactoring linter
linting typescript javascript kotlin golang nodejs svelte flutter nextjs vite
sideload sideloaded sideloading adb apk apks emulator? oled amoled nvme ssd gpu
gpus tpu cpu? hdmi usb vpn dns ssl tls url urls uri udp tcp ipv6 gigabyte
terabyte petabyte firmware middleware crypto bitcoin ethereum blockchain nft
stablecoin defi hodl staking sats mah ghz mhz hz nits qi
ibuprofen paracetamol acetaminophen tylenol advil amoxicillin antihistamine
melatonin probiotic probiotics electrolyte electrolytes cortisol dopamine
serotonin triglycerides sciatica tendonitis tinnitus apnea telehealth copay
dermatologist cardiologist neurologist pediatrician physiotherapy physio
chiropractor mri ekg ecg covid vaccine? booster?""".replace("?", "").split()

# Brand and acronym casing the caps table should enforce (exact canonical forms;
# ambiguous readings like python/java/rust/react/swift/windows stay out on the same
# rule as the name holdouts).
CASED = {
    "qwen": "Qwen", "chatgpt": "ChatGPT", "gpt": "GPT", "deepseek": "DeepSeek",
    "mistral": "Mistral", "anthropic": "Anthropic", "openai": "OpenAI",
    "midjourney": "Midjourney", "kubernetes": "Kubernetes", "postgres": "Postgres",
    "postgresql": "PostgreSQL", "graphql": "GraphQL", "github": "GitHub",
    "gitlab": "GitLab", "typescript": "TypeScript", "javascript": "JavaScript",
    "kotlin": "Kotlin", "nodejs": "NodeJS", "svelte": "Svelte", "flutter": "Flutter",
    "json": "JSON", "yaml": "YAML", "toml": "TOML", "oled": "OLED", "amoled": "AMOLED",
    "nvme": "NVMe", "ssd": "SSD", "gpu": "GPU", "tpu": "TPU", "hdmi": "HDMI",
    "usb": "USB", "vpn": "VPN", "dns": "DNS", "ssl": "SSL", "tls": "TLS",
    "url": "URL", "urls": "URLs", "udp": "UDP", "tcp": "TCP", "ipv6": "IPv6",
    "nft": "NFT", "bitcoin": "Bitcoin", "ethereum": "Ethereum", "tylenol": "Tylenol",
    "advil": "Advil", "mri": "MRI", "ekg": "EKG", "ecg": "ECG", "ollama": "Ollama",
}

vocab = [w.rstrip("\n") for w in open(VOCAB, encoding="utf-8")]
have = set(vocab)
caps_lines = open(CAPS, encoding="utf-8").read().splitlines()
caps_have = set(l.split("\t")[0] for l in caps_lines if l and not l.startswith("#") and "\t" in l)

cased_names = sorted((set(NAMES) | set(PLACES)) - PLACE_HOLDOUTS)
new_words = sorted(set(w for w in SLANG + TECH if w not in have)) + [n for n in cased_names if n not in have]
if len(vocab) + len(new_words) > ID_CAP:
    sys.exit("id space exhausted: %d + %d > %d" % (len(vocab), len(new_words), ID_CAP))

if new_words:
    with open(VOCAB, "a", encoding="utf-8") as f:
        for w in new_words:
            f.write(w + "\n")

caps_entries = {n: n.capitalize() for n in cased_names}
caps_entries.update(CASED)
new_caps = sorted(k for k in caps_entries if k not in caps_have)
if new_caps:
    with open(CAPS, "a", encoding="utf-8") as f:
        if CAPS_MARK not in caps_lines:
            f.write(CAPS_MARK + "\n")
        for n in new_caps:
            f.write("%s\t%s\n" % (n, caps_entries[n]))

print("words appended: %d (vocab now %d)" % (len(new_words), len(vocab) + len(new_words)))
print("caps entries added: %d" % len(new_caps))
