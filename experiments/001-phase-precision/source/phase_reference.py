"""Integer reference for the two E001 phase-rotation cores (no ML dependencies)."""
import math


def round_even_ratio(numerator, denominator):
    """Round an exact signed rational to nearest integer, with ties to even."""
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or
                          (2 * remainder == denominator and quotient & 1))


def coefficients():
    return ([round(math.cos(i * math.tau / 256) * 127) for i in range(256)],
            [round(math.sin(i * math.tau / 256) * 127) for i in range(256)])


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def quantized_rotation(phase, increment, x, y, acc_bits, cosine, sine):
    phase = (phase + increment) & ((1 << acc_bits) - 1)
    address = round_even_ratio(phase, 256) & 255 if acc_bits == 16 else phase
    real = x * cosine[address] - y * sine[address]
    imag = x * sine[address] + y * cosine[address]
    assert -(1 << 24) <= real < (1 << 24)
    assert -(1 << 24) <= imag < (1 << 24)
    return phase, address, real, imag


def pack_vector(step16, step8, x, y):
    return ((step16 << 40) | (step8 << 32) |
            ((x & 65535) << 16) | (y & 65535))


def unpack_vector(word):
    return (word >> 40, (word >> 32) & 255,
            signed(word >> 16, 16), signed(word, 16))


def pack_golden(guard, low):
    _, ag, rg, ig = guard
    _, al, rl, il = low
    mask = (1 << 25) - 1
    return ((rg & mask) | ((ig & mask) << 25) | ((rl & mask) << 50) |
            ((il & mask) << 75) | (ag << 100) | (al << 108))
