##                                                        

# Om prosjektet



Studentene velger selv prosjektoppgave. Se under for foreslåtte problemstillinger.

Målet med prosjektoppgaven er at dere får erfaring med å løse et eller flere gitte praktiske problemstillinger ved hjelp av maskinlæring. Metodene dere bruker kan være de dere har lært i undervisningen/øvingene, eller nye metoder dere setter dere inn i. Noen av oppgaveforslagene kan også være litteraturstudier der dere oppsummerer hva som har blitt gjort tidligere innenfor et gitt fagfelt. Oppgaven du velger kan også fungere som forarbeid til Bacheloroppgaven, og om dette er aktuelt diskuter dette med Donn eller Ole.

Dere får karakter ut i fra jobben dere gjør, og selv om dere ikke oppnår ønsket resultat så er det likevel mulig å oppnå karakteren A. Med andre ord, noen problemstillinger kan være vanskelig å løse på en tilfredsstillende måte, men letingen etter løsningen og konklusjonen fra resultatene kan fortsatt være svært interessant.

- Gruppestørrelse: 1-4 studenter
- Oppstart: 24.10
- Presentasjon: kommer senere
- Innlevering: 19.11, klokken 12:00 på Inspera
  - Artikkel på maks 2000 ord
    - Figurer, tabeller, referanser og liknende ekskludert
    - Typisk struktur: Abstract, Introduction, Related Work, Methods, Results and Discussion, Conclusion and Future Work
      - Strukturen kan variere med oppgavetype
    - Ta utgangspunkt i at leseren tar/har tatt dette faget (IDATT2502)
    - Kan bruke latex-templaten i blackboard mappen for prosjektet
    - Artikkelen kan skrives enten på engelsk eller norsk.
  - Artikkel med eventuell kildekode leveres på Inspera som en komprimert fil
- Møter med veileder hver/annenhver uke

# Prosjektoppgaver



Velg en av oppgavene nedenfor, og send oppgave-tittel med eventuell prosjektbeskrivelse og deltakere (fullt navn og epost) til [ole.c.eidheim@ntnu.no](mailto:ole.c.eidheim@ntnu.no).

## Egendefinert oppgave



Veileder: tildelt etter kompetanse

Dere kan i samarbeid med en veileder (Ole, Donn, Pedro eller Olav) lage deres egen oppgave.

For inspirasjon til å finne en motiverende prosjektoppgave, kan dere se på tilgjengelige datasett, for eksempel:

- https://www.kaggle.com/datasets
- https://toolbox.google.com/datasetsearch
- https://huggingface.co/datasets

I tilfellet dere vil utforske andre maskinlæringsmetoder som et forprosjekt til en spesifikk bachelor-oppgave innen maskinlæring, kan posteren også skrives som en review article.

## Visualisation of binary programs from different instruction sets



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

Use methods for visualising high-dimensional data on the cpu_rec and ISAdetect datasets. t-SNE (https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) can be used.

- https://github.com/airbus-seclab/cpu_rec
- https://paperswithcode.com/dataset/isadetect-dataset

Experiment with feature sets, for example, byte frequency distributions, bigram and trigram frequency distributions, etc.

### Project aim



What can we learn about the instruction set architecture (ISA) or binary program through visualisation? Does the same program across different architectures have characteristics in common? Which ISAs have commonalities?

### Suggested reading



- https://www.sstic.org/2017/presentation/cpu_rec/
- https://arxiv.org/pdf/2204.06624.pdf
- https://dl.acm.org/doi/abs/10.1145/3374664.3375742

## Endianness detection from binary programs



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

Use supervised learning to build a model that can identify the endianness of an input binary program. Train the model using ISAdetect while holding one architecture out for testing. Cross-validation will help with model evaluation. Use the ISAdetect dataset (or a subset of it). Hold each architecture out in turn as the test set while training the model on the remaining architectures.

### Project aim



Can we reliably detect endianness from binary programs where instruction set architecture information is unknown?

### Suggested reading



- https://dl.acm.org/doi/abs/10.1145/3374664.3375742

## Ad block for podcasts



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

Use speech recognition and a large language model (e.g., the ChatGPT API or a local instance) to convert podcast audio to text and then classify sentences as either advert or non-advert. This method can be unsupervised, but you may need to create a small test set to measure accuracy.

### Project aim



Ad block for podcasts lags behind ad blocking technology (e.g., the uBlock Origin browser add-on) on the web because the audio medium is more challenging. Several podcast platforms like Acast dynamically insert adverts based on IP geolocation. Use ML methods to help solve this problem and give end-users an ad-free podcast experience. Build a working demo! It is suggested that the demo be implemented as an HTTP intercepting proxy, based on the mitmproxy project.

### Suggested reading



- https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3004164?show=full&locale-attribute=en
- https://www.adblockradio.com/en/

## Malware classification using the EMBER dataset



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

- Compare against baseline approach
- Emphasis on simpler ML models over complex
- https://arxiv.org/abs/1804.04637

## Benchmarking models on a simulated mushroom dataset



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

- Create a ML pipeline comparing many models
- https://www.nature.com/articles/s41598-021-87602-3
- https://github.com/ghattab/secondarydata

## Captcha solving with image understanding: Literature review and case study



Supervisor: Donn Morrison [donn.morrison@ntnu.no](mailto:donn.morrison@ntnu.no)

- Google Gemini currently proves fairly capable of automatically solving hCAPTCHA puzzles
- Conduct a literature review
- Possibly implement a demo
- https://accounts.hcaptcha.com/demo

## Generative AI: Implementing a VAE, Diffusion Model, or GAN



Supervisor: Olav Finne Præsteng Larsen [olav.f.p.larsen@ntnu.no](mailto:olav.f.p.larsen@ntnu.no)

### Project Description



In this project, you will explore **Generative AI**, a rapidly growing field of machine learning focused on creating models that can generate new data from learned distributions. You will choose and implement one of the following models:

1. **Variational Autoencoder (VAE)**: A type of generative model that learns latent representations of data and can generate new samples by sampling from this learned latent space.
2. **Diffusion Model**: A probabilistic model that learns to reverse the process of adding noise to data, producing high-quality samples.
3. **Generative Adversarial Network (GAN)**: A two-part neural network where a generator creates fake samples, and a discriminator evaluates their authenticity.

#### Key Objectives:



- Implement a VAE, Diffusion Model, or GAN from scratch using a deep learning framework like **TensorFlow** or **PyTorch**.
- Train the model on a dataset such as **MNIST**, **CIFAR-10**, or any custom dataset of your choice.
- Evaluate the performance (IS, FID) of the model by generating and visualizing new samples

## NP-problemer vs. Maskinlæring: En Utforskende Analyse



Veileder: Pedro Pablo Cardona Arroyave [pedro.p.cardona@ntnu.no](mailto:pedro.p.cardona@ntnu.no)

### Bakgrunn



NP-problemer (Non-deterministic Polynomial-time) representerer en klasse av problemer i informatikken som er spesielt utfordrende å løse effektivt med tradisjonelle algoritmiske tilnærminger. Disse problemene er kjent for at verifikasjonen av en løsning kan gjøres relativt raskt, men å finne løsningen kan være ekstremt tidkrevende, spesielt ved større datasett.

Maskinlæring, derimot, tilbyr potensielt nye måter å tilnærme seg disse problemene på ved å lære fra data heller enn å stole på en fast algoritme.

### Oppgave



I denne analysen skal det sammenlignes to metoder for å løse et velkjent NP-problem: en tradisjonell algoritme og en maskinlæringsmodell. Spesielt kan det fokuseres på reisende selger-problemet (Travelling Salesman Problem, TSP) eller åtte dronninger-problemet (Eight Queens Puzzle), men andre NP-problemer kan også utforskes.

#### Implementering



1. **Tradisjonell algoritme**: Implementer en løsning på NP-problemet ved hjelp av en tradisjonell algoritme. Den kan variere avhengig av problemet, for eksempel en genetisk algoritme for TSP eller en tilbakeføring for åtte dronninger-problemet.
2. **Maskinlæringsmodell**: Bruk en maskinlæringsbasert tilnærming som forsterkende læring (Reinforcement Learning) eller nevrale nettverk (Neural Networks) for å forsøke å forutsi optimale løsninger for problemet.

### Sammenligning av metoder



Sammenlign resultatene fra de to metodene med tanke på:

- **Nøyaktighet**: Hvor god løsningen er i forhold til den optimale løsningen.
- **Tidsforbruk**: Hvor lang tid hver metode tar å fullføre, spesielt ved større problemstørrelser.
- **Skalerbarhet**: Hvordan hver metode håndterer økende problemstørrelser.
- **Kompleksitet**: Hvor enkelt det er å implementere og tilpasse hver metode, spesielt maskinlæringsmodellen.

### Videre arbeid



- **Visualisering**: Lag en visuell sammenligning av løsningene for å bedre forstå forskjellene mellom metodene.
- **Optimalisering**: Forsøk å optimalisere hver metode for å forbedre resultatene ytterligere.
- **Generalisering**: Utforsk hvordan hver metode kan generaliseres til andre NP-problemer og datasett.

### Konklusjon



Denne analysen har som mål å utforske om maskinlæring kan være en effektiv tilnærming til å løse NP-problemer, sammenlignet med tradisjonelle algoritmiske metoder. Resultatene vil kunne bidra til videre forskning på hvordan maskinlæring kan anvendes i optimeringsproblemer innenfor NP-klassen.

## Lære av feil: oppnå tilfredsstillende agenter med færre episoder



Veileder: Ole Christian Eidheim [ole.c.eidheim@ntnu.no](mailto:ole.c.eidheim@ntnu.no)

I denne oppgaven skal dere eksperimentere med observasjons-rommet direkte i reinforcement learning miljøer som [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) og [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/). Utgangspunktet er å lagre observasjoner som førte til at agenten feilet og bruke disse observasjonene til å ta andre handlinger slik at feilene unngås ved senere episoder. Målet er å oppnå tilfredsstillende agenter gjennom færre episoder enn det som er typisk med tradisjonelle reinforcement learning metoder.

Ta gjerne utgangspunkt i følgende eksempelkode:

```
import gymnasium
import pygame
import numpy as np

render_mode = "human"  # Set to None to run without graphics

env = gymnasium.make("CartPole-v1", render_mode=render_mode)
env.action_space.seed(0)
np.random.seed(0)

observation, info = env.reset(seed=0)

steps_alive = 0
terminated_observations = np.zeros((0, 4))
terminated_observations_normalized = np.zeros((0, 4))
mean = np.zeros((4))
std = np.ones((4))
last_closest_distance_change = None
action = 0
while True:
    last_closest_distance = np.min(
        np.sum(np.square((observation - mean) / std - terminated_observations_normalized), axis=1)
    ) if terminated_observations_normalized.shape[0] > 0 else 0.0
    observation, reward, terminated, truncated, info = env.step(action)

    closest_distance = np.min(
        np.sum(np.square((observation - mean) / std - terminated_observations_normalized), axis=1)
    ) if terminated_observations.shape[0] > 0 else 0.0

    closest_distance_change = closest_distance - last_closest_distance

    if closest_distance_change >= 0.0 or (
        last_closest_distance_change != None and closest_distance_change > last_closest_distance_change
    ):
        pass
    else:
        action = (action + 1) % 2

    last_closest_distance_change = closest_distance_change

    steps_alive += 1

    if np.abs(observation[2]) > 1.0 or np.abs(observation[0]) > 2.4 or truncated:
        if not truncated:
            terminated_observations = np.concatenate((terminated_observations, [observation]), axis=0)
            mean = terminated_observations.mean(axis=0)
            std = terminated_observations.std(axis=0)
            std = np.where(std != 0., std, 1.)  # To avoid division by zero
            terminated_observations_normalized = (terminated_observations - mean) / std

        print(f"Steps alive: {steps_alive}")
        steps_alive = 0
        last_closest_distance_change = None
        observation, info = env.reset()
```

## Diverse Reinforcement Learning oppgaver



Veileder: Ole Christian Eidheim [ole.c.eidheim@ntnu.no](mailto:ole.c.eidheim@ntnu.no)

Oppgaveforslag:

- Velg et environment (for eksempel fra [Gymnasium](https://gymnasium.farama.org/)) - test forskjellige algoritmer
- Velg en algoritme - test forskjellige environment
- Lag et environment
- Utforske forskjellige konfigurasjoner av en algoritme

 
