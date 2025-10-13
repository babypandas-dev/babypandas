import babypandas as bpd

df1 = bpd.DataFrame(
    **{
        "data": {
            "letter": ["a", "b", "c"],
            "count": [9, 3, 3],
            "idx": [0, 1, 2],
        }
    }
)

print(df1)
