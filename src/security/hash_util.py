import base64
import hashlib


def verify_hash(plain: str, stored_b64: str) -> bool:
    try:
        decoded = base64.b64decode(stored_b64).decode("utf-8").strip()
        algorithm, iterations, salt, expected_hex = decoded.split("$", 3)
        iterations = int(iterations)

        algo = algorithm.lower().replace("-", "")
        plain = plain.strip()

        # 等价 Java：MessageDigest digest = getInstance(...)
        h = hashlib.new(algo)

        data = (salt + plain).encode("utf-8")

        for _ in range(iterations):
            h.update(data)
            data = h.digest()
            h = hashlib.new(algo) 

        return data.hex() == expected_hex

    except Exception as e:
        return False
