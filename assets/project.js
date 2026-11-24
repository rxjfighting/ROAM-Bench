"use strict";
window.ROAM_PROJECT = {
  authors: [
    {name: "Xingjian Ran", affiliations: [1, 4]},
    {name: "Xiaoye Mo", affiliations: [2]},
    {name: "Sihao Liu", affiliations: [1]},
    {name: "Li Luo", affiliations: [1]},
    {name: "Taiqiang Wu", affiliations: [1, 4]},
    {name: "Ji Zhao", affiliations: [3, 4]},
    {name: "Wenqi Shao", affiliations: [4]},
    {name: "Hongyang Li", affiliations: [4], marker: "‡"},
    {name: "Jiaqi Wang", affiliations: [4], marker: "†"},
    {name: "Bo Dai", affiliations: [1]}
  ],
  affiliations: [
    "The University of Hong Kong",
    "University of Electronic Science and Technology of China",
    "Shanghai Innovation Institute",
    "JD.com"
  ],
  resources: {
    paper: "https://arxiv.org/pdf/2611.33143",
    code: "https://github.com/rxjfighting/ROAM-Bench",
    scenes: "https://huggingface.co/datasets/xjran/ROAM-Bench-scenes"
  },
  groups: [
    {code: "G1", count: 49, title: "Semantic-spatial grounding",
      description: "Find the right object or room by connecting visual appearance, room function, and location.",
      types: ["Surface contents", "Room localization"],
      example: "What is on this surface? Explore to locate it and inspect its contents.",
      illustrations: [
        {image: "surface-contents", title: "Surface contents", question: "What rests on this surface?", alt: "A counter with a small black device, shown with a detail inset."},
        {image: "room-localization", title: "Room localization", question: "Which room contains the glass?", alt: "A glass on a wooden counter, shown with a detail inset."}
      ]},
    {code: "G2", count: 38, title: "Spatial aggregation",
      description: "Combine observations across rooms and views, keeping track of distinct objects and the area already searched.",
      types: ["Object count", "Room count"],
      example: "Search the home and count the target objects, including those in closed storage.",
      illustrations: [
        {image: "object-count", title: "Object count", question: "How many mirrors are in the house?", alt: "Mirrors visible in a room during a whole-home counting task."},
        {image: "room-count", title: "Room count", question: "How many rooms contain smoke detectors?", alt: "A room observation used to check for smoke detectors."}
      ]},
    {code: "G3", count: 77, title: "Metric and comparative geometry",
      description: "Estimate physical sizes and distances, or compare which object is nearer, using the views collected during exploration.",
      types: ["Object distance", "Room area", "Gap width", "Nearer-object comparison"],
      example: "Which of these two objects is closer to the reference object?",
      illustrations: [
        {image: "object-distance", title: "Object distance", question: "How far apart are the two objects?", diagram: true, alt: "A table and storage shelf in an object-distance task."},
        {image: "nearer-object", title: "Nearer-object comparison", question: "Which object is closest to the shoe bench?", diagram: true, alt: "The shoe bench used as a reference for comparing object proximity."},
        {image: "room-area", title: "Room area", question: "What is the reception room's floor area?", diagram: true, alt: "A camera observation of the reception room."},
        {image: "gap-width", title: "Gap width", question: "How wide is the clear gap between them?", diagram: true, alt: "The gap between a cabinet and seating, marked with a double-ended arrow."}
      ]},
    {code: "G4", count: 56, title: "Reference-frame integration",
      description: "Judge directions from a task-defined viewpoint, or relate the starting position to the final position and heading.",
      types: ["Relative direction", "Path integration"],
      example: "After the instructed exploration, which direction leads back to the starting position?",
      illustrations: [
        {image: "relative-direction", title: "Relative direction", question: "Face the wardrobe from the nightstand. Which way is the shoe bench?", diagram: true, alt: "A bedroom view for a direction judgment in the task-defined reference frame."},
        {image: "path-integration", title: "Path integration", question: "From here, which way is the start?", diagram: true, alt: "An episode observation used to judge the direction back to the starting position."}
      ]},
    {code: "G5", count: 30, title: "Goal-directed spatial action",
      description: "Find the specified object, pick that instance, and transfer it to the designated destination.",
      types: ["Relocation"],
      example: "Find the seed sample jar, pick it up, and put it on the laboratory bench.",
      stages: true,
      illustrations: [
        {image: "relocation-target", title: "Locate the target", alt: "The source jar on a counter before pickup."},
        {image: "relocation-destination", title: "Reach the destination", alt: "The laboratory bench before placement."},
        {image: "relocation-placed", title: "Observe the placed object", alt: "The same jar after placement on the laboratory bench."}
      ]}
  ],
  results: [
    {model: "GPT-5.6-sol", gap: 38, tasks: 250, A: 34.4, T: 19.2, N: 38.0, S: 31.2, errorS: 47.4, actions: 74.11},
    {model: "GPT-6.1-sol", gap: 30, tasks: 250, A: 58.8, T: 46.8, N: 67.6, S: 56.8, errorS: 31.0, actions: 50.82},
    {model: "DeepSeek-V4.1-Flash", gap: 22, tasks: 250, A: 19.2, T: 10.4, N: 32.0, S: 23.6, errorS: 72.9, actions: 66.27},
    {model: "Claude Sonnet 5.5", gap: 31, tasks: 250, A: 44.0, T: 31.6, N: 53.2, S: 41.2, errorS: 39.8, actions: 60.30},
    {model: "Kimi K3.1 Pro", gap: 39, tasks: 250, A: 44.0, T: 28.4, N: 55.2, S: 46.4, errorS: 45.7, actions: 79.10},
    {model: "Claude Opus 5.5", gap: 36, tasks: 250, A: 54.4, T: 40.0, N: 63.2, S: 52.8, errorS: 34.1, actions: 53.18},
    {model: "GPT-6-astra", gap: 10, tasks: 250, A: 73.2, T: 69.2, N: 83.6, S: 68.0, errorS: 15.9, actions: 43.62},
    {model: "Human", gap: 0, tasks: 100, A: 84.0, T: 84.0, N: 92.0, S: 73.0, errorS: 4.1, actions: 53.55, human: true}
  ]
};
