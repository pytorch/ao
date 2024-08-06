import torch

"""
Contains generic functions to pack and unpack uintx (2-7) tensors into uint8 tensors.
"""


def down_size_uint2(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4)


def up_size_uint2(size):
    return (*size[:-1], size[-1] * 4)


def unpack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
    # since we are using uint8 we will decode 4 entries per byte
    shape = uint8_data.shape
    uint8_data = uint8_data.to(torch.uint8)
    first_elements = (uint8_data >> 6) & 0b11
    second_elements = (uint8_data >> 4) & 0b11
    third_elements = (uint8_data >> 2) & 0b11
    fourth_elements = uint8_data & 0b11
    return torch.stack(
        (first_elements, second_elements, third_elements, fourth_elements), dim=-1
    ).view(up_size_uint2(shape))


def pack_uint2(uint8_data: torch.Tensor) -> torch.Tensor:
    """pack lowest 2 bits of 2 uint8 -> 1 uint8"""
    shape = uint8_data.shape
    assert shape[-1] % 4 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (
        (uint8_data[::4] & 0b11) << 6
        | (uint8_data[1::4] & 0b11) << 4
        | (uint8_data[2::4] & 0b11) << 2
        | (uint8_data[3::4] & 0b11)
    ).view(down_size_uint2(shape))
    return packed_data


def down_size_uint3(size):
    assert size[-1] % 8 == 0, f"{size} last dim not divisible by eight"
    return (*size[:-1], size[-1] // 8 * 3)


def up_size_uint3(size):
    assert size[-1] % 3 == 0, f"{size} last dim not divisible by three"
    return (*size[:-1], size[-1] // 3 * 8)


def unpack_uint3(uint8_data: torch.Tensor) -> torch.Tensor:
    """
    3 -> 8
    01234567|01234567|01234567
    AAABBBCC|CDDDEEEF|FFGGGHHH
    """
    shape = uint8_data.shape
    uint8_data = uint8_data.to(torch.uint8)

    return torch.stack(
        (
            (uint8_data[::3] >> 5) & 0b111,
            (uint8_data[::3] >> 2) & 0b111,
            (uint8_data[::3] & 0b11) << 1 | (uint8_data[1::3] >> 7) & 0b1,
            (uint8_data[1::3] >> 4) & 0b111,
            (uint8_data[1::3] >> 1) & 0b111,
            (uint8_data[1::3] & 0b1) << 2 | (uint8_data[2::3] >> 6) & 0b11,
            (uint8_data[2::3] >> 3) & 0b111,
            uint8_data[2::3] & 0b111,
        ),
        dim=-1,
    ).view(up_size_uint3(shape))


def pack_uint3(uint8_data: torch.Tensor) -> torch.Tensor:
    """
    8 -> 3
    01234567|01234567|01234567
    AAABBBCC|CDDDEEEF|FFGGGHHH
    """

    shape = uint8_data.shape
    assert shape[-1] % 8 == 0
    uint8_data = uint8_data.contiguous().view(-1)

    packed_data = torch.stack(
        (
            (
                (uint8_data[::8] & 0b111) << 5
                | (uint8_data[1::8] & 0b111) << 2
                | (uint8_data[2::8] & 0b111) >> 1
            ),
            (
                (uint8_data[2::8] & 0b1) << 7
                | (uint8_data[3::8] & 0b111) << 4
                | (uint8_data[4::8] & 0b111) << 1
                | ((uint8_data[5::8] >> 2) & 1)
            ),
            (
                (uint8_data[5::8] & 0b11) << 6
                | (uint8_data[6::8] & 0b111) << 3
                | (uint8_data[7::8] & 0b111)
            ),
        ),
        dim=-1,
    ).view(down_size_uint3(shape))

    return packed_data


def down_size_uint4(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def up_size_uint4(size):
    return (*size[:-1], size[-1] * 2)


def unpack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    uint8_data = uint8_data.to(torch.uint8)
    first_elements = (uint8_data >> 4) & 0b1111
    second_elements = uint8_data & 0b1111
    return torch.stack((first_elements, second_elements), dim=-1).view(
        up_size_uint4(shape)
    )


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    packed_data = (uint8_data[::2] << 4 | (uint8_data[1::2] & 0b1111)).view(
        down_size_uint4(shape)
    )
    return packed_data


def down_size_uint5(size):
    assert size[-1] % 8 == 0, f"{size} last dim not divisible by 8"
    return (*size[:-1], size[-1] // 8 * 5)


def up_size_uint5(size):
    assert size[-1] % 5 == 0, f"{size} last dim not divisible by 5"
    return (*size[:-1], size[-1] // 5 * 8)


def pack_uint5(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack the 5 lowest bits of 8 input bytes into 5 bytes

    8 -> 5
    01234567|01234567|01234567|01234567|01234567
    AAAAABBB|BBCCCCCD|DDDDEEEE|EFFFFFGG|GGGHHHHH

    The packing pattern:
    - First byte:  (A0 A1 A2 A3 A4 B0 B1 B2)
    - Second byte: (B3 B4 C0 C1 C2 C3 C4 D0)
    - Third byte:  (D1 D2 D3 D4 E0 E1 E2 E3)
    - Fourth byte: (E4 F0 F1 F2 F3 F4 G0 G1)
    - Fifth byte:  (G2 G3 G4 H0 H1 H2 H3 H4)
    """
    shape = uint8_data.shape
    assert (
        shape[-1] % 8 == 0
    ), f"Input last dimension should be divisible by 8, but got {shape[-1]}"

    uint8_data = uint8_data.contiguous().view(-1, 8)

    packed_data = torch.stack(
        (
            ((uint8_data[:, 0] & 0b00011111) << 3)
            | ((uint8_data[:, 1] & 0b00011100) >> 2),
            ((uint8_data[:, 1] & 0b00000011) << 6)
            | ((uint8_data[:, 2] & 0b00011111) << 1)
            | ((uint8_data[:, 3] & 0b10000) >> 4),
            ((uint8_data[:, 3] & 0b00001111) << 4)
            | ((uint8_data[:, 4] & 0b00011110) >> 1),
            ((uint8_data[:, 4] & 0b00000001) << 7)
            | ((uint8_data[:, 5] & 0b00011111) << 2)
            | ((uint8_data[:, 6] & 0b0011000) >> 3),
            ((uint8_data[:, 6] & 0b00000111) << 5) | (uint8_data[:, 7] & 0b00011111),
        ),
        dim=-1,
    ).view(down_size_uint5(shape))

    return packed_data


def unpack_uint5(packed_data: torch.Tensor) -> torch.Tensor:
    """Unpack the 5 bytes into the 5 lowest bits of 8 bytes
    01234567|01234567|01234567|01234567|01234567
    AAAAABBB|BBCCCCCD|DDDDEEEE|EFFFFFGG|GGGHHHHH
    """
    shape = packed_data.shape
    assert (
        shape[-1] % 5 == 0
    ), f"Input last dimension should be divisible by 5, but got {shape[-1]}"

    packed_data = packed_data.contiguous().view(-1, 5)

    unpacked_data = torch.stack(
        (
            ((packed_data[:, 0] >> 3) & 0b00011111),
            ((packed_data[:, 0] & 0b00000111) << 2)
            | ((packed_data[:, 1] >> 6) & 0b00000011),
            ((packed_data[:, 1] >> 1) & 0b00011111),
            ((packed_data[:, 1] & 0b00000001) << 4)
            | ((packed_data[:, 2] >> 4) & 0b00001111),
            ((packed_data[:, 2] & 0b00001111) << 1)
            | ((packed_data[:, 3] >> 7) & 0b00000001),
            ((packed_data[:, 3] >> 2) & 0b00011111),
            ((packed_data[:, 3] & 0b00000011) << 3)
            | ((packed_data[:, 4] >> 5) & 0b00000111),
            packed_data[:, 4] & 0b00011111,
        ),
        dim=-1,
    ).view(up_size_uint5(shape))

    return unpacked_data


def down_size_uint6(size):
    assert size[-1] % 4 == 0, f"{size} last dim not divisible by four"
    return (*size[:-1], size[-1] // 4 * 3)


def up_size_uint6(size):
    assert size[-1] % 3 == 0, f"{size} last dim not divisible by three"
    return (*size[:-1], size[-1] // 3 * 4)


def pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack the 6 lowest bits of 4 input bytes into 3 bytes

    4 -> 3
    01234567|01234567|01234567
    AAAAAABB|BBBBCCCC|CCDDDDDD

    The packing pattern:
    - First byte:  (A0 A1 A2 A3 A4 A5 B0 B1)
    - Second byte: (B2 B3 B4 B5 C0 C1 C2 C3)
    - Third byte:  (C4 C5 D0 D1 D2 D3 D4 D5)
    """
    shape = uint8_data.shape
    assert (
        shape[-1] % 4 == 0
    ), f"Input last dimension should be divisible by 4, but got {shape[-1]}"

    uint8_data = uint8_data.contiguous().view(-1, 4)

    packed_data = torch.stack(
        (
            ((uint8_data[:, 0] & 0b00111111) << 2)
            | ((uint8_data[:, 1] >> 4) & 0b00000011),
            ((uint8_data[:, 1] & 0b00001111) << 4)
            | ((uint8_data[:, 2] >> 2) & 0b00001111),
            ((uint8_data[:, 2] & 0b00000011) << 6) | (uint8_data[:, 3] & 0b00111111),
        ),
        dim=-1,
    ).view(down_size_uint6(shape))

    return packed_data


def unpack_uint6(packed_data: torch.Tensor) -> torch.Tensor:
    """Unpack the 3 bytes into the 6 lowest bits of 4 outputs
    01234567|01234567|01234567
    AAAAAABB|BBBBCCCC|CCDDDDDD
    """
    shape = packed_data.shape
    assert (
        shape[-1] % 3 == 0
    ), f"Input last dimension should be divisible by 3, but got {shape[-1]}"

    packed_data = packed_data.contiguous().view(-1, 3)

    unpacked_data = torch.stack(
        (
            (packed_data[:, 0] >> 2) & 0b00111111,
            ((packed_data[:, 0] & 0b00000011) << 4)
            | ((packed_data[:, 1] >> 4) & 0b00001111),
            ((packed_data[:, 1] & 0b00001111) << 2)
            | ((packed_data[:, 2] >> 6) & 0b00000011),
            packed_data[:, 2] & 0b00111111,
        ),
        dim=-1,
    ).view(up_size_uint6(shape))

    return unpacked_data


def down_size_uint7(size):
    assert size[-1] % 8 == 0, f"{size} last dim not divisible by 8"
    return (*size[:-1], size[-1] // 8 * 7)


def up_size_uint7(size):
    assert size[-1] % 7 == 0, f"{size} last dim not divisible by 7"
    return (*size[:-1], size[-1] // 7 * 8)


def pack_uint7(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack the 7 lowest bits of 8 input bytes into 7 bytes

    8 -> 7
    01234567|01234567|01234567|01234567|01234567|01234567|01234567
    AAAAAAAB|BBBBBBCC|CCCCCDDD|DDDDEEEE|EEEFFFFF|FFGGGGGG|GHHHHHHH

    The packing pattern:
    - First byte:  (A0 A1 A2 A3 A4 A5 A6 B0)
    - Second byte: (B1 B2 B3 B4 B5 B6 C0 C1)
    - Third byte:  (C2 C3 C4 C5 C6 D0 D1 D2)
    - Fourth byte: (D3 D4 D5 D6 E0 E1 E2 E3)
    - Fifth byte:  (E4 E5 E6 F0 F1 F2 F3 F4)
    - Sixth byte:  (F5 F6 G0 G1 G2 G3 G4 G5)
    - Seventh byte:(G6 H0 H1 H2 H3 H4 H5 H6)
    """
    shape = uint8_data.shape
    assert (
        shape[-1] % 8 == 0
    ), f"Input last dimension should be divisible by 8, but got {shape[-1]}"

    uint8_data = uint8_data.contiguous().view(-1, 8)

    packed_data = torch.stack(
        (
            ((uint8_data[:, 0] & 0b01111111) << 1)
            | ((uint8_data[:, 1] >> 6) & 0b00000001),
            ((uint8_data[:, 1] & 0b00111111) << 2)
            | ((uint8_data[:, 2] >> 5) & 0b00000011),
            ((uint8_data[:, 2] & 0b00011111) << 3)
            | ((uint8_data[:, 3] >> 4) & 0b00000111),
            ((uint8_data[:, 3] & 0b00001111) << 4)
            | ((uint8_data[:, 4] >> 3) & 0b00001111),
            ((uint8_data[:, 4] & 0b00000111) << 5)
            | ((uint8_data[:, 5] >> 2) & 0b00011111),
            ((uint8_data[:, 5] & 0b00000011) << 6)
            | ((uint8_data[:, 6] >> 1) & 0b00111111),
            ((uint8_data[:, 6] & 0b00000001) << 7)
            | ((uint8_data[:, 7] >> 0) & 0b01111111),
        ),
        dim=-1,
    ).view(down_size_uint7(shape))

    return packed_data


def unpack_uint7(packed_data: torch.Tensor) -> torch.Tensor:
    """Unpack the 7 bytes into the 7 lowest bits of 8 bytes
    01234567|01234567|01234567|01234567|01234567|01234567|01234567
    AAAAAAAB|BBBBBBCC|CCCCCDDD|DDDDEEEE|EEEFFFFF|FFGGGGGG|GHHHHHHH
    """
    shape = packed_data.shape
    assert (
        shape[-1] % 7 == 0
    ), f"Input last dimension should be divisible by 7, but got {shape[-1]}"

    packed_data = packed_data.contiguous().view(-1, 7)

    unpacked_data = torch.stack(
        (
            (packed_data[:, 0] >> 1) & 0b01111111,
            ((packed_data[:, 0] & 0b00000001) << 6)
            | ((packed_data[:, 1] >> 2) & 0b01111111),
            ((packed_data[:, 1] & 0b00000011) << 5)
            | ((packed_data[:, 2] >> 3) & 0b01111111),
            ((packed_data[:, 2] & 0b00000111) << 4)
            | ((packed_data[:, 3] >> 4) & 0b01111111),
            ((packed_data[:, 3] & 0b00001111) << 3)
            | ((packed_data[:, 4] >> 5) & 0b01111111),
            ((packed_data[:, 4] & 0b00011111) << 2)
            | ((packed_data[:, 5] >> 6) & 0b01111111),
            ((packed_data[:, 5] & 0b00111111) << 1)
            | ((packed_data[:, 6] >> 7) & 0b01111111),
            packed_data[:, 6] & 0b01111111,
        ),
        dim=-1,
    ).view(up_size_uint7(shape))

    return unpacked_data
