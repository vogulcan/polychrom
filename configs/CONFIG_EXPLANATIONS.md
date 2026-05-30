Config1:
Active transcription that should reflect biological relevance with respect to nascent RNA production rates and enhancer-promoter biology, and for constitutive transcription. It should also reflect 3D genome and transcription relevance, where RNAPII can act as a semi-permeable barrier to loop extrusion.

Config2:
Transcription is shut down.

Expected differences between Config1 and Config2:
1. In Config1, there will be active transcription occurring, which means it is more likely that RNAPII can act as a barrier to loop extrusion, potentially leading to higher rates of cohesin at gene bodies, rather than CTCF boundaries, thus config1 should result in shorter loop lengths than the config2. Because of this, TADs should be less defined in config1 than in config2.
2. Even though, boundary strength is slightly higher in config1 than in config2, the overall TAD structure should be more defined in config2 than in config1, because of the lack of transcription and the fact that cohesin will be more likely to accumulate at CTCF boundaries rather than gene bodies. Improved corner scores in config2 should reflect this. Thus, TAD strenght (intra-TAD/inter-TAD interactions) should be higher in config2 than in config1.
3. Since, boundary strength is slightly higher in config1 than in config2, the results in config2 should lead to more prominent stripe formation that are boundary crossing.
Key takeaways:
Config2 should have larger loops leading to more defined TADs with higher corner scores, but more boundary crossing stripes.